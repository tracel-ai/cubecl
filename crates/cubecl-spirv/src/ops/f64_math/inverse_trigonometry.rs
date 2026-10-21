//! `asin`, `acos`, `atan` and `atan2` to 3.5 ulp, from SLEEF `xasin`, `xacos`, `xatan` and
//! `xatan2`.

use cubecl_core::{self as cubecl, prelude::*};

use super::base::{FRAC_PI_2, FRAC_PI_4, PI, PI_HI, PI_LO, mul_sign, poly, sign_bit};
use super::double_double::{add_df, dd};

const ASIN_COEFFICIENTS: [f64; 12] = [
    0.031_615_876_506_539_346,
    -0.015_819_182_433_299_966,
    0.019_290_454_772_679_107,
    0.006_606_077_476_277_171,
    0.012_153_605_255_773_773,
    0.013_887_151_845_016_092,
    0.017_359_569_912_236_146,
    0.022_371_761_819_320_483,
    0.030_381_959_280_381_322,
    0.044_642_856_813_771_024,
    0.075_000_000_003_785_82,
    0.166_666_666_666_649_75,
];

const ATAN_COEFFICIENTS: [f64; 19] = [
    -1.887_960_084_630_735e-5,
    0.000_209_850_076_645_816_98,
    -0.001_106_118_314_866_724_8,
    0.003_700_267_441_887_131,
    -0.008_898_961_958_876_555,
    0.016_599_329_773_529_202,
    -0.025_451_762_493_231_264,
    0.033_785_258_000_135_31,
    -0.040_762_919_127_683_65,
    0.046_666_715_007_784_06,
    -0.052_367_485_230_348_246,
    0.058_766_639_292_667_36,
    -0.066_657_357_936_108_05,
    0.076_921_953_831_176_96,
    -0.090_908_995_008_245_01,
    0.111_111_105_648_261_42,
    -0.142_857_142_667_713_3,
    0.199_999_999_996_591_27,
    -0.333_333_333_333_311_1,
];

#[cube]
pub(super) fn asin(d: f64) -> f64 {
    let small = d.abs() < 0.5;
    let x2 = select(small, d * d, (1.0 - d.abs()) * 0.5);
    let x = select(small, d.abs(), x2.sqrt());
    let u = fma(poly(x2, ASIN_COEFFICIENTS), x * x2, x);
    let r = select(small, u, fma(u, -2.0, FRAC_PI_2));
    mul_sign(r, d)
}

#[cube]
pub(super) fn acos(d: f64) -> f64 {
    let small = d.abs() < 0.5;
    let x2 = select(small, d * d, (1.0 - d.abs()) * 0.5);
    let x = select(small, d.abs(), x2.sqrt());
    let x = select(d.abs() == 1.0, 0.0, x);
    let u = poly(x2, ASIN_COEFFICIENTS) * (x2 * x);

    let near_zero = FRAC_PI_2 - (mul_sign(x, d) + mul_sign(u, d));
    let result = select(small, near_zero, (x + u) * 2.0);
    let from_pi = add_df(dd(PI_HI, PI_LO), -result).hi;
    select(!small && d < 0.0, from_pi, result)
}

#[cube]
pub(super) fn atan(x: f64) -> f64 {
    let magnitude = x.abs();
    let inverted = 1.0 < magnitude;
    let reduced = select(inverted, 1.0 / magnitude, magnitude);

    let square = reduced * reduced;
    let angle = fma(reduced, square * poly(square, ATAN_COEFFICIENTS), reduced);

    let angle = select(inverted, FRAC_PI_2 - angle, angle);
    select(sign_bit(x), -angle, angle)
}

#[cube]
pub(super) fn atan2(y: f64, x: f64) -> f64 {
    let r = mul_sign(atan2k(y.abs(), x), x);
    let x_edge = select(x.is_inf(), mul_sign(FRAC_PI_2, x), 0.0);
    let r = select(x.is_inf() || x == 0.0, FRAC_PI_2 - x_edge, r);
    let y_edge = select(x.is_inf(), mul_sign(FRAC_PI_4, x), 0.0);
    let r = select(y.is_inf(), FRAC_PI_2 - y_edge, r);
    let r = select(y == 0.0, select(sign_bit(x), PI, 0.0), r);
    select(x.is_nan() || y.is_nan(), f64::NAN, mul_sign(r, y))
}

/// `atan2` for `y >= 0`, before the special cases.
#[cube]
fn atan2k(y: f64, x: f64) -> f64 {
    let mut quadrant = select(sign_bit(x), -2i32, 0);
    let x = x.abs();
    let steep = x < y;
    quadrant = select(steep, quadrant + 1, quadrant);
    let numerator = select(steep, -x, y);
    let denominator = select(steep, y, x);

    let ratio = numerator / denominator;
    let square = ratio * ratio;
    let angle = fma(ratio, square * poly(square, ATAN_COEFFICIENTS), ratio);
    fma(f64::cast_from(quadrant), FRAC_PI_2, angle)
}
