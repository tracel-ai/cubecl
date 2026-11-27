//! Bit access and helpers shared by the f64 functions, from SLEEF `commonfuncs.h`.

use cubecl_core::{self as cubecl, prelude::*};

pub(super) const PI: f64 = core::f64::consts::PI;
pub(super) const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
pub(super) const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;
pub(super) const FRAC_1_PI: f64 = core::f64::consts::FRAC_1_PI;
pub(super) const FRAC_2_PI: f64 = core::f64::consts::FRAC_2_PI;

/// pi as a double-double: the head is the f64 nearest pi and the tail is the rest.
pub(super) const PI_HI: f64 = PI;
pub(super) const PI_LO: f64 = 1.224_646_799_147_353_2e-16;

/// ln 2 as a double-double.
pub(super) const LN2_HI: f64 = core::f64::consts::LN_2;
pub(super) const LN2_LO: f64 = 2.319_046_813_846_299_6e-17;

/// ln 2 split for range reduction: the head has trailing zero bits, so `k * L2U` is exact.
pub(super) const L2U: f64 = 0.693_147_180_559_663;
pub(super) const L2L: f64 = 2.823_529_056_303_157_7e-13;
pub(super) const R_LN2: f64 = core::f64::consts::LOG2_E;

/// The largest input of `exp` with a finite result.
pub(super) const LOG_DBL_MAX: f64 = 709.782_712_893_384;
pub(super) const SQRT_DBL_MAX: f64 = 1.340_780_792_994_259_6e154;
pub(super) const DBL_MIN: f64 = f64::MIN_POSITIVE;

/// The low and high 32-bit words of `x`.
#[cube]
pub(super) fn words(x: f64) -> Vector<u32, Const<2>> {
    Vector::<u32, Const<2>>::reinterpret(x)
}

#[cube]
pub(super) fn high_word(x: f64) -> u32 {
    words(x).extract(1usize)
}

#[cube]
pub(super) fn from_words(low: u32, high: u32) -> f64 {
    let mut words = Vector::<u32, Const<2>>::empty();
    words.insert(0usize, low);
    words.insert(1usize, high);
    f64::reinterpret(words)
}

#[cube]
pub(super) fn sign_bit(x: f64) -> bool {
    (high_word(x) >> 31) == 1
}

/// `x` with its sign flipped when `y` has its sign bit set.
#[cube]
pub(super) fn mul_sign(x: f64, y: f64) -> f64 {
    let sign = high_word(y) & 0x8000_0000;
    let x_words = words(x);
    from_words(x_words.extract(0usize), x_words.extract(1usize) ^ sign)
}

/// 2^q, for q in [-1022, 1023].
#[cube]
pub(super) fn pow2i(q: i32) -> f64 {
    from_words(0, u32::cast_from(q + 0x3ff) << 20)
}

/// `d * 2^e` for e in [-2044, 2046], which rounds once when the result is subnormal.
#[cube]
pub(super) fn ldexp2(d: f64, e: i32) -> f64 {
    let half = e >> 1;
    d * pow2i(half) * pow2i(e - half)
}

/// `d * 2^q` by adding to the exponent field, for a normal `d` and a normal result.
#[cube]
pub(super) fn ldexp3(d: f64, q: i32) -> f64 {
    let d_words = words(d);
    let high = u32::cast_from(i32::cast_from(d_words.extract(1usize)) + (q << 20));
    from_words(d_words.extract(0usize), high)
}

/// The unbiased exponent of a positive `d`, subnormals included.
#[cube]
pub(super) fn ilogbk(d: f64) -> i32 {
    let small = d < 4.909_093_465_297_727e-91;
    let d = select(small, 2.037_035_976_334_486e90 * d, d);
    let biased = i32::cast_from((high_word(d) >> 20) & 0x7ff);
    biased - select(small, 300 + 0x3ff, 0x3ff)
}

/// The unbiased exponent field of a normal `d`.
#[cube]
pub(super) fn ilogb2k(d: f64) -> i32 {
    i32::cast_from((high_word(d) >> 20) & 0x7ff) - 0x3ff
}

#[cube]
pub(super) fn is_int(d: f64) -> bool {
    d.round() == d
}

#[cube]
pub(super) fn is_odd(d: f64) -> bool {
    let half = d * 0.5;
    half.round() != half
}

/// A polynomial in `x` by Horner's rule. The coefficients go from the highest power down, in
/// the order SLEEF lists them.
#[cube]
pub(super) fn poly<const D: usize>(x: f64, #[comptime] coefficients: [f64; D]) -> f64 {
    let mut total = f64::new(coefficients[0]);

    #[unroll]
    for i in 1..D {
        total = fma(total, x, f64::new(coefficients[comptime![i]]));
    }

    total
}
