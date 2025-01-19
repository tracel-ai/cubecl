use cubecl_core as cubecl;
use cubecl_core::{prelude::*, CubeLaunch};

#[derive(CubeLaunch)]
pub struct FastDivmod<I: Int> {
    divisor: I,
    multiplier: u32,
    shift_right: u32,
}

impl<I: Int> FastDivmod<I> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<'a, R: Runtime>(divisor: I) -> FastDivmodLaunch<'a, I, R> {
        let div_int = divisor.to_i64().unwrap();
        assert!(div_int != 0);

        let mut multiplier = 0;
        let mut shift_right = 0;

        if div_int != 1 {
            let p = 31 + find_log2(div_int);
            multiplier = (1u64 << p).div_ceil(div_int as u64) as u64;
            shift_right = p - 32;
        }

        FastDivmodLaunch::new(
            ScalarArg::new(divisor),
            ScalarArg::new(multiplier as u32),
            ScalarArg::new(shift_right as u32),
        )
    }
}

impl<I: Int> FastDivmod<I> {
    pub fn div(&self, dividend: I) -> I {
        self.div_mod(dividend).0
    }

    pub fn modulo(&self, dividend: I) -> I {
        self.div_mod(dividend).1
    }

    pub fn div_mod(&self, dividend: I) -> (I, I) {
        (dividend / self.divisor, dividend % self.divisor)
    }

    pub fn __expand_div(
        context: &mut CubeContext,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        this.__expand_div_method(context, dividend)
    }

    pub fn __expand_modulo(
        context: &mut CubeContext,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        this.__expand_modulo_method(context, dividend)
    }

    pub fn __expand_div_mod(
        context: &mut CubeContext,
        this: FastDivmodExpand<I>,
        dividend: ExpandElementTyped<I>,
    ) -> (ExpandElementTyped<I>, ExpandElementTyped<I>) {
        this.__expand_div_mod_method(context, dividend)
    }
}

impl<I: Int> FastDivmodExpand<I> {
    pub fn __expand_div_method(
        self,
        context: &mut CubeContext,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        self.__expand_div_mod_method(context, dividend).0
    }

    pub fn __expand_modulo_method(
        self,
        context: &mut CubeContext,
        dividend: ExpandElementTyped<I>,
    ) -> ExpandElementTyped<I> {
        self.__expand_div_mod_method(context, dividend).1
    }

    pub fn __expand_div_mod_method(
        self,
        context: &mut CubeContext,
        dividend: ExpandElementTyped<I>,
    ) -> (ExpandElementTyped<I>, ExpandElementTyped<I>) {
        fast_divmod::expand::<I>(
            context,
            dividend,
            self.divisor,
            self.multiplier,
            self.shift_right,
        )
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

fn find_log2(x: i64) -> i64 {
    let mut a = (31 - x.leading_zeros()) as i64;
    a += ((x & (x - 1)) != 0) as i64;
    a
}
