use cubecl_core::prelude::*;
use cubecl_core::{
    self as cubecl,
    ir::{ElemType, UIntKind},
};
use cubecl_runtime::TypeUsage;

/// Create a fast-divmod object if supported, or a regular fallback if not.
/// This precalculates certain values on the host, in exchange for making division and modulo
/// operations on the GPU much faster. Only supports u32 right now to allow for a simpler algorithm.
/// It's mostly used for indices regardless.
///
/// Implementation based on ONNX:
/// <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h>
#[derive(CubeType, Clone, Copy)]
pub enum FastDivmod<I: Int + ScalarArgSettings = u32> {
    Fast {
        divisor: I,
        multiplier: I,
        shift_right: u32,
    },
    Fallback {
        divisor: I,
    },
}

impl FastDivmodArgs<u32> {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, divisor: u32) -> Self {
        debug_assert!(divisor != 0);

        if !u64::supported_uses(client).contains(TypeUsage::Arithmetic) {
            return FastDivmodArgs::Fallback {
                divisor: ScalarArg::new(divisor),
            };
        }

        let (shift, multiplier) = find_params_u32(divisor);

        FastDivmodArgs::Fast {
            divisor: ScalarArg::new(divisor),
            multiplier: ScalarArg::new(multiplier),
            shift_right: ScalarArg::new(shift),
        }
    }
}

impl FastDivmodArgs<usize> {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, divisor: usize) -> Self {
        debug_assert!(divisor != 0);

        if !u64::supported_uses(client).contains(TypeUsage::Arithmetic) {
            return FastDivmodArgs::Fallback {
                divisor: ScalarArg::new(divisor),
            };
        }

        FastDivmodArgs::UsizeUninit {
            divisor: ScalarArg::new(divisor),
        }
    }
}

#[cube]
impl<I: Int + ScalarArgSettings + MulHi> FastDivmod<I> {
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
    pub enum FastDivmodArgs<I: Int + ScalarArgSettings = u32> {
        Fast {
            divisor: ScalarArg<I>,
            multiplier: ScalarArg<I>,
            shift_right: ScalarArg<u32>,
        },
        Fallback {
            divisor: ScalarArg<I>,
        },
        UsizeUninit {
            divisor: ScalarArg<I>,
        },
    }

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub enum FastDivmodCompilationArg<I: Int + ScalarArgSettings> {
        Fast {
            divisor: ScalarCompilationArg<I>,
            multiplier: ScalarCompilationArg<I>,
            shift_right: ScalarCompilationArg<u32>,
        },
        Fallback {
            divisor: ScalarCompilationArg<I>,
        },
    }

    impl<I: Int + ScalarArgSettings> CompilationArg for FastDivmodCompilationArg<I> {}

    impl<I: Int + ScalarArgSettings, R: Runtime> ArgSettings<R> for FastDivmodArgs<I> {
        fn register(&self, launcher: &mut KernelLauncher<R>) {
            match self {
                FastDivmodArgs::Fast {
                    divisor,
                    multiplier,
                    shift_right,
                } => {
                    divisor.register(launcher);
                    multiplier.register(launcher);
                    shift_right.register(launcher);
                }
                FastDivmodArgs::Fallback { divisor } => {
                    divisor.register(launcher);
                }
                FastDivmodArgs::UsizeUninit { divisor } => {
                    match launcher.settings.address_type.unsigned_type().elem_type() {
                        ElemType::UInt(UIntKind::U32) => {
                            let (shift, multiplier) =
                                find_params_u32(divisor.elem.to_u32().unwrap());
                            ScalarArg::new(shift).register(launcher);
                            ScalarArg::new(I::from_int(multiplier as i64)).register(launcher);
                        }
                        ElemType::UInt(UIntKind::U64) => {
                            let (shift, multiplier) =
                                find_params_u64(divisor.elem.to_u64().unwrap());
                            ScalarArg::new(shift).register(launcher);
                            ScalarArg::new(I::from_int(multiplier as i64)).register(launcher);
                        }
                        other => panic!("FastDivmod doesn't support address type {other:?}"),
                    }
                }
            }
        }
    }

    impl<I: Int + ScalarArgSettings> LaunchArg for FastDivmod<I> {
        type RuntimeArg<'a, R: Runtime> = FastDivmodArgs<I>;
        type CompilationArg = FastDivmodCompilationArg<I>;

        fn compilation_arg<'a, R: Runtime>(
            runtime_arg: &Self::RuntimeArg<'a, R>,
        ) -> Self::CompilationArg {
            match runtime_arg {
                FastDivmodArgs::Fast { .. } | FastDivmodArgs::UsizeUninit { .. } => {
                    FastDivmodCompilationArg::Fast {
                        divisor: ScalarCompilationArg::new(),
                        multiplier: ScalarCompilationArg::new(),
                        shift_right: ScalarCompilationArg::new(),
                    }
                }
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
