use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Create a fast-divmod object if supported, or a regular fallback if not.
/// This precalculates certain values on the host, in exchange for making division and modulo
/// operations on the GPU much faster. Only supports u32 right now to allow for a simpler algorithm.
/// It's mostly used for indices regardless.
///
/// Implementation based on ONNX:
/// <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h>
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub enum FastDivmod {
    Fast {
        divisor: u32,
        multiplier: u32,
        shift_right: u32,
    },
    PowerOfTwo {
        shift: u32,
        mask: u32,
    },
    Fallback {
        divisor: u32,
    },
}

impl<R: Runtime> FastDivmodArgs<'_, R> {
    pub fn new(client: &ComputeClient<R::Server, R::Channel>, divisor: u32) -> Self {
        debug_assert!(divisor != 0);

        if divisor.is_power_of_two() {
            return FastDivmodArgs::PowerOfTwo {
                shift: ScalarArg::new(divisor.trailing_zeros()),
                mask: ScalarArg::new(divisor - 1),
            };
        }

        if !u64::is_supported(client) {
            return FastDivmodArgs::Fallback {
                divisor: ScalarArg::new(divisor),
            };
        }

        let div_64 = divisor as u64;
        let shift = find_log2(divisor);
        let multiplier = ((1u64 << 32) * ((1u64 << shift) - div_64)) / div_64 + 1;

        FastDivmodArgs::Fast {
            divisor: ScalarArg::new(divisor),
            multiplier: ScalarArg::new(multiplier as u32),
            shift_right: ScalarArg::new(shift as u32),
        }
    }
}

#[cube]
impl FastDivmod {
    pub fn div(&self, dividend: u32) -> u32 {
        match self {
            FastDivmod::Fast {
                multiplier,
                shift_right,
                ..
            } => {
                let t = u32::mul_hi(dividend, *multiplier);
                (t + dividend) >> shift_right
            }
            FastDivmod::PowerOfTwo { shift, .. } => dividend >> *shift,
            FastDivmod::Fallback { divisor } => dividend / divisor,
        }
    }

    pub fn modulo(&self, dividend: u32) -> u32 {
        let q = self.div(dividend);
        match self {
            FastDivmod::Fast { divisor, .. } => dividend - q * divisor,
            FastDivmod::PowerOfTwo { mask, .. } => dividend & mask,
            FastDivmod::Fallback { divisor } => dividend % divisor,
        }
    }

    pub fn div_mod(&self, dividend: u32) -> (u32, u32) {
        let q = self.div(dividend);
        let r = match self {
            FastDivmod::Fast { divisor, .. } => dividend - q * divisor,
            FastDivmod::Fallback { divisor } => dividend - q * divisor,
            FastDivmod::PowerOfTwo { mask, .. } => dividend & *mask,
        };

        (q, r)
    }
}

fn find_log2(x: u32) -> usize {
    for i in 0..32 {
        if (1 << i) >= x {
            return i;
        }
    }
    32
}
