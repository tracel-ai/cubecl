use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, ir::ConstantValue};
use cubecl_runtime::TypeUsage;

/// Create a fast-divmod object if supported, or a regular fallback if not.
/// This precalculates certain values on the host, in exchange for making division and modulo
/// operations on the GPU much faster. Only supports u32 right now to allow for a simpler algorithm.
/// It's mostly used for indices regardless.
///
/// Implementation based on ONNX:
/// <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h>
#[derive(CubeType, Clone, Copy)]
pub enum FastDivmod<I: FastDivmodInt> {
    Fast {
        divisor: I,
        multiplier: I,
        shift_right: u32,
    },
    Fallback {
        divisor: I,
    },
}

pub trait FastDivmodInt: Int + MulHi + ScalarArgSettings {
    fn size<R: Runtime>(launcher: &KernelLauncher<R>) -> usize;
}

// Could potentially support signed, but that needs more handling

impl FastDivmodInt for u32 {
    fn size<R: Runtime>(_launcher: &KernelLauncher<R>) -> usize {
        size_of::<u32>()
    }
}

impl FastDivmodInt for usize {
    fn size<R: Runtime>(launcher: &KernelLauncher<R>) -> usize {
        launcher.settings.address_type.unsigned_type().size()
    }
}

impl<I: FastDivmodInt> FastDivmodArgs<I> {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, divisor: I) -> Self {
        debug_assert!({
            let divisor_value: ConstantValue = divisor.into();
            let divisor_value = divisor_value.as_u64();
            divisor_value != 0
        });

        if !u64::supported_uses(client).contains(TypeUsage::Arithmetic) {
            return FastDivmodArgs::Fallback {
                divisor: ScalarArg::new(divisor),
            };
        }

        FastDivmodArgs::Fast {
            divisor: ScalarArg::new(divisor),
        }
    }
}

#[cube]
impl<I: FastDivmodInt> FastDivmod<I> {
    pub fn div(&self, dividend: I) -> I {
        match self {
            FastDivmod::Fast {
                multiplier,
                shift_right,
                ..
            } => {
                let t = I::mul_hi(dividend, *multiplier);
                (t + dividend) >> I::cast_from(*shift_right)
            }
            FastDivmod::Fallback { divisor } => dividend / *divisor,
        }
    }

    pub fn modulo(&self, dividend: I) -> I {
        let q = self.div(dividend);
        match self {
            FastDivmod::Fast { divisor, .. } => dividend - q * *divisor,
            FastDivmod::Fallback { divisor } => dividend % *divisor,
        }
    }

    pub fn div_mod(&self, dividend: I) -> (I, I) {
        let q = self.div(dividend);
        let r = match self {
            FastDivmod::Fast { divisor, .. } => dividend - q * *divisor,
            FastDivmod::Fallback { divisor } => dividend - q * *divisor,
        };

        (q, r)
    }
}

fn find_params_u32(divisor: u32) -> (u32, u32) {
    let div_64 = divisor as u64;
    let shift = divisor.next_power_of_two().trailing_zeros();
    let multiplier = ((1u64 << 32) * ((1u64 << shift) - div_64)) / div_64 + 1;
    (shift, multiplier as u32)
}

fn find_params_u64(divisor: u64) -> (u32, u64) {
    let div_128 = divisor as u128;
    let shift = divisor.next_power_of_two().trailing_zeros();
    let multiplier = ((1u128 << 64) * ((1u128 << shift) - div_128)) / div_128 + 1;
    (shift, multiplier as u64)
}

mod launch {
    use super::*;

    #[derive(Clone, Copy)]
    pub enum FastDivmodArgs<I: FastDivmodInt = usize> {
        Fast { divisor: ScalarArg<I> },
        Fallback { divisor: ScalarArg<I> },
    }

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub enum FastDivmodCompilationArg<I: FastDivmodInt> {
        Fast {
            divisor: ScalarCompilationArg<I>,
            multiplier: ScalarCompilationArg<I>,
            shift_right: ScalarCompilationArg<u32>,
        },
        Fallback {
            divisor: ScalarCompilationArg<I>,
        },
    }

    impl<I: FastDivmodInt> CompilationArg for FastDivmodCompilationArg<I> {}

    impl<I: FastDivmodInt, R: Runtime> ArgSettings<R> for FastDivmodArgs<I> {
        fn register(&self, launcher: &mut KernelLauncher<R>) {
            match self {
                FastDivmodArgs::Fast { divisor } => {
                    let (shift_right, multiplier) = match <I as FastDivmodInt>::size(launcher) {
                        4 => {
                            let divisor = divisor.elem.to_u32().unwrap();
                            let (shift, multiplier) = find_params_u32(divisor);

                            let shift = ScalarArg::new(shift);
                            let multiplier = ScalarArg::new(I::from_int(multiplier as i64));
                            (shift, multiplier)
                        }
                        8 => {
                            let divisor = divisor.elem.to_u64().unwrap();
                            let (shift, multiplier) = find_params_u64(divisor);

                            let shift = ScalarArg::new(shift);
                            let multiplier = ScalarArg::new(I::from_int(multiplier as i64));
                            (shift, multiplier)
                        }
                        _ => panic!("unsupported type size for FastDivmod"),
                    };
                    divisor.register(launcher);
                    multiplier.register(launcher);
                    shift_right.register(launcher);
                }
                FastDivmodArgs::Fallback { divisor } => {
                    divisor.register(launcher);
                }
            }
        }
    }

    impl<I: FastDivmodInt> LaunchArg for FastDivmod<I> {
        type RuntimeArg<'a, R: Runtime> = FastDivmodArgs<I>;
        type CompilationArg = FastDivmodCompilationArg<I>;

        fn compilation_arg<'a, R: Runtime>(
            runtime_arg: &Self::RuntimeArg<'a, R>,
        ) -> Self::CompilationArg {
            match runtime_arg {
                FastDivmodArgs::Fast { .. } => FastDivmodCompilationArg::Fast {
                    divisor: ScalarCompilationArg::new(),
                    multiplier: ScalarCompilationArg::new(),
                    shift_right: ScalarCompilationArg::new(),
                },
                FastDivmodArgs::Fallback { .. } => FastDivmodCompilationArg::Fallback {
                    divisor: ScalarCompilationArg::new(),
                },
            }
        }

        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut cubecl::prelude::KernelBuilder,
        ) -> <Self as cubecl::prelude::CubeType>::ExpandType {
            match arg {
                FastDivmodCompilationArg::Fast {
                    divisor,
                    multiplier,
                    shift_right,
                } => FastDivmodExpand::Fast {
                    divisor: I::expand(divisor, builder),
                    multiplier: I::expand(multiplier, builder),
                    shift_right: u32::expand(shift_right, builder),
                },
                FastDivmodCompilationArg::Fallback { divisor } => FastDivmodExpand::Fallback {
                    divisor: I::expand(divisor, builder),
                },
            }
        }

        fn expand_output(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            Self::expand(arg, builder)
        }
    }
}
pub use launch::FastDivmodArgs;
