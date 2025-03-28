use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Create a fast-divmod object if supported, or a regular fallback if not.
/// This precalculates certain values on the host, in exchange for making division and modulo
/// operations on the GPU much faster.
#[derive(CubeType, CubeLaunch)]
pub enum FastDivmod<I: Int + MulHi> {
    Fast {
        divisor: I,
        #[allow(unused)]
        multiplier: u32,
        #[allow(unused)]
        shift_right: u32,
    },
    Fallback {
        divisor: I,
    },
}

impl<I: Int + MulHi, R: Runtime> FastDivmodArgs<'_, I, R> {
    pub fn new(client: &ComputeClient<R::Server, R::Channel>, divisor: I) -> Self {
        if !u64::is_supported(client) {
            return FastDivmodArgs::Fallback {
                divisor: ScalarArg::new(divisor),
            };
        }
        let div_int = divisor.to_i64().unwrap();
        assert!(div_int != 0);

        let mut multiplier = 0;
        let mut shift_right = 0;

        if div_int != 1 {
            let p = 31 + find_log2(div_int);
            multiplier = (1u64 << p).div_ceil(div_int as u64) as u64;
            shift_right = p - 32;
        }

        FastDivmodArgs::Fast {
            divisor: ScalarArg::new(divisor),
            multiplier: ScalarArg::new(multiplier as u32),
            shift_right: ScalarArg::new(shift_right as u32),
        }
    }
}

impl<I: Int + MulHi> FastDivmod<I> {
    pub fn div(&self, dividend: I) -> I {
        self.div_mod(dividend).0
    }

    pub fn modulo(&self, dividend: I) -> I {
        self.div_mod(dividend).1
    }

    pub fn div_mod(&self, dividend: I) -> (I, I) {
        let divisor = match self {
            FastDivmod::Fast { divisor, .. } => *divisor,
            FastDivmod::Fallback { divisor } => *divisor,
        };
        (dividend / divisor, dividend % divisor)
    }

    pub fn __expand_div(
        context: &mut Scope,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        this.__expand_div_method(context, dividend)
    }

    pub fn __expand_modulo(
        context: &mut Scope,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        this.__expand_modulo_method(context, dividend)
    }

    pub fn __expand_div_mod(
        context: &mut Scope,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> (ExpandElementTyped<I>, ExpandElementTyped<I>) {
        this.__expand_div_mod_method(context, dividend)
    }
}

impl<I: Int + MulHi> FastDivmodExpand<I> {
    pub fn __expand_div_method(
        self,
        context: &mut Scope,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        self.__expand_div_mod_method(context, dividend).0
    }

    pub fn __expand_modulo_method(
        self,
        context: &mut Scope,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        self.__expand_div_mod_method(context, dividend).1
    }

    pub fn __expand_div_mod_method(
        self,
        context: &mut Scope,
        dividend: ExpandElementTyped<I>,
    ) -> (ExpandElementTyped<I>, ExpandElementTyped<I>) {
        match self {
            FastDivmodExpand::Fast {
                divisor,
                multiplier,
                shift_right,
            } => fast_divmod::expand::<I>(context, dividend, divisor, multiplier, shift_right),
            FastDivmodExpand::Fallback { divisor } => {
                divmod::expand::<I>(context, dividend, divisor)
            }
        }
    }
}

#[cube]
pub fn fast_divmod<I: Int>(dividend: I, divisor: I, multiplier: u32, shift_right: u32) -> (I, I) {
    let quotient = if divisor != I::new(1) {
        I::cast_from(u32::mul_hi(u32::cast_from(dividend), multiplier) >> shift_right)
    } else {
        dividend
    };

    let remainder = dividend - (quotient * divisor);
    (quotient, remainder)
}

#[cube]
pub fn divmod<I: Int>(dividend: I, divisor: I) -> (I, I) {
    let quotient = dividend / divisor;
    let remainder = dividend % divisor;
    (quotient, remainder)
}

fn find_log2(x: i64) -> i64 {
    let mut a = (31 - x.leading_zeros()) as i64;
    a += ((x & (x - 1)) != 0) as i64;
    a
}
