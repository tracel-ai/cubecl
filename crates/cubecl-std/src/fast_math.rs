use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

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
    // A zero divisor arises only when a tensor has a zero-sized dimension
    // (e.g. Brush's `sh_coeffs_rest` of shape [N, 0, 3] at SH degree 0).
    // Such tensors cause 0 workgroups to be dispatched, so these params are
    // never read by any kernel thread — return a dummy pair instead of
    // panicking during kernel launch preparation.
    if divisor == 0 {
        return (0, 0);
    }
    let div_64 = divisor as u64;
    let shift = divisor.next_power_of_two().trailing_zeros();
    let multiplier = ((1u64 << 32) * ((1u64 << shift) - div_64)) / div_64 + 1;
    (shift, multiplier as u32)
}

fn find_params_u64(divisor: u64) -> (u32, u64) {
    if divisor == 0 {
        return (0, 0);
    }
    let div_128 = divisor as u128;
    let shift = divisor.next_power_of_two().trailing_zeros();
    let multiplier = ((1u128 << 64) * ((1u128 << shift) - div_128)) / div_128 + 1;
    (shift, multiplier as u64)
}

mod launch {
    use cubecl_core::ir::UIntKind;

    use super::*;

    #[derive_cube_comptime]
    pub enum FastDivmodCompilationArg {
        Fast,
        Fallback,
    }

    impl<I: FastDivmodInt> LaunchArg for FastDivmod<I> {
        type RuntimeArg<R: Runtime> = I;
        type CompilationArg = FastDivmodCompilationArg;

        fn register<R: Runtime>(
            divisor: Self::RuntimeArg<R>,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            let props = launcher.with_scope(|scope| scope.properties.clone().unwrap());
            let fast = props.features.supports_type(UIntKind::U64);
            match fast {
                true => {
                    let (shift_right, multiplier) = match <I as FastDivmodInt>::size(launcher) {
                        4 => {
                            let divisor = divisor.to_u32().unwrap();
                            let (shift, multiplier) = find_params_u32(divisor);

                            let multiplier = I::from_int(multiplier as i64);
                            (shift, multiplier)
                        }
                        8 => {
                            let divisor = divisor.to_u64().unwrap();
                            let (shift, multiplier) = find_params_u64(divisor);

                            let multiplier = I::from_int(multiplier as i64);
                            (shift, multiplier)
                        }
                        _ => panic!("unsupported type size for FastDivmod"),
                    };
                    <I as LaunchArg>::register(divisor, launcher);
                    <I as LaunchArg>::register(multiplier, launcher);
                    <u32 as LaunchArg>::register(shift_right, launcher);
                    FastDivmodCompilationArg::Fast
                }
                false => {
                    <I as LaunchArg>::register(divisor, launcher);
                    FastDivmodCompilationArg::Fallback
                }
            }
        }

        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut cubecl::prelude::KernelBuilder,
        ) -> <Self as cubecl::prelude::CubeType>::ExpandType {
            match arg {
                FastDivmodCompilationArg::Fast => FastDivmodExpand::Fast {
                    divisor: I::expand(&(), builder),
                    multiplier: I::expand(&(), builder),
                    shift_right: u32::expand(&(), builder),
                },
                FastDivmodCompilationArg::Fallback => FastDivmodExpand::Fallback {
                    divisor: I::expand(&(), builder),
                },
            }
        }
    }
}
