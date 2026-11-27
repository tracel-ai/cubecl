//! `exp` and `pow` to 1 ulp and `log` to 3.5 ulp, from SLEEF `xexp`, `xpow` and `xlog`, with the
//! kernels the hyperbolic functions share.

use cubecl_core::{self as cubecl, prelude::*};

use super::base::{
    DBL_MIN, L2L, L2U, LN2_HI, LN2_LO, LOG_DBL_MAX, R_LN2, ilogb2k, ilogbk, is_int, is_odd, ldexp2,
    ldexp3, mul_sign, poly, sign_bit,
};
use super::double_double::{
    Dd, add_dd, add_fd, add2_df, add2_ff, dd, div_dd, mul_dd, mul_df, normalize, scale, square,
};

/// 2^64, to bring a subnormal argument into the normal range.
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

const EXP_COEFFICIENTS: [f64; 10] = [
    2.081_276_378_237_164_5e-9,
    2.511_210_703_042_288e-8,
    2.755_762_628_169_491e-7,
    2.755_723_402_025_388_2e-6,
    2.480_158_687_479_686_3e-5,
    0.000_198_412_698_985_586_58,
    0.001_388_888_888_914_497_8,
    0.008_333_333_333_314_938,
    0.041_666_666_666_666_026,
    0.166_666_666_666_666_9,
];

const EXPM1_COEFFICIENTS: [f64; 10] = [
    2.088_606_211_072_837e-9,
    2.511_129_308_928_765_2e-8,
    2.755_739_112_349_004_7e-7,
    2.755_723_629_119_288_3e-6,
    2.480_158_715_923_547_3e-5,
    0.000_198_412_698_960_509_2,
    0.001_388_888_888_897_745,
    0.008_333_333_333_316_527,
    0.041_666_666_666_666_505,
    0.166_666_666_666_666_85,
];

const EXPK_COEFFICIENTS: [f64; 10] = [
    2.510_696_834_209_504_2e-8,
    2.762_861_667_702_706_5e-7,
    2.755_724_967_250_235_7e-6,
    2.480_149_739_898_198e-5,
    0.000_198_412_698_809_069_8,
    0.001_388_888_893_997_713,
    0.008_333_333_333_323_714,
    0.041_666_666_666_540_95,
    0.166_666_666_666_666_74,
    0.500_000_000_000_001,
];

const LOG_COEFFICIENTS: [f64; 7] = [
    0.153_487_338_491_425_07,
    0.152_519_917_006_351_95,
    0.181_863_266_251_982_99,
    0.222_221_366_518_767_37,
    0.285_714_294_746_548,
    0.399_999_999_950_799_6,
    0.666_666_666_666_777_9,
];

const LOGK_COEFFICIENTS: [f64; 9] = [
    0.116_255_524_079_935_04,
    0.103_239_680_901_072_95,
    0.117_754_809_412_464,
    0.133_329_810_868_462_74,
    0.153_846_227_114_512_26,
    0.181_818_180_850_050_78,
    0.222_222_222_230_083_56,
    0.285_714_285_714_249_17,
    0.400_000_000_000_000_1,
];
const LOGK_TWO_THIRDS_HI: f64 = 0.666_666_666_666_666_6;
const LOGK_TWO_THIRDS_LO: f64 = 3.805_549_625_424_120_6e-17;

const LOGK2_COEFFICIENTS: [f64; 7] = [
    0.138_604_363_904_671_68,
    0.131_699_838_841_615_37,
    0.153_914_168_346_271_95,
    0.181_816_523_941_564_6,
    0.222_222_246_326_620_35,
    0.285_714_285_511_134_1,
    0.400_000_000_000_914,
];
const LOGK2_TWO_THIRDS: f64 = 0.666_666_666_666_664_9;

#[cube]
pub(super) fn exp(d: f64) -> f64 {
    let quotient = (d * R_LN2).round();
    let exponent = i32::cast_from(quotient);
    let reduced = fma(quotient, -L2U, d);
    let reduced = fma(quotient, -L2L, reduced);

    let series = poly(reduced, EXP_COEFFICIENTS);
    let series = fma(series, reduced, 0.5);
    let series = fma(series, reduced, 1.0);
    let series = fma(series, reduced, 1.0);
    let result = ldexp2(series, exponent);

    let result = select(d > LOG_DBL_MAX, f64::INFINITY, result);
    select(d < -1000.0, 0.0, result)
}

/// `exp(d) - 1`, accurate near zero.
#[cube]
pub(super) fn expm1k(d: f64) -> f64 {
    let quotient = (d * R_LN2).round();
    let exponent = i32::cast_from(quotient);
    let reduced = fma(quotient, -L2U, d);
    let reduced = fma(quotient, -L2L, reduced);

    let square = reduced * reduced;
    let series = poly(reduced, EXPM1_COEFFICIENTS);
    let series = fma(square, 0.5, square * reduced * series) + reduced;
    let scaled = ldexp2(series + 1.0, exponent) - 1.0;
    select(exponent == 0, series, scaled)
}

#[cube]
pub(super) fn log(d: f64) -> f64 {
    let subnormal = d < DBL_MIN;
    let scaled = select(subnormal, d * TWO_POW_64, d);
    let exponent = ilogb2k(scaled * (1.0 / 0.75));
    let mantissa = ldexp3(scaled, -exponent);
    let exponent = select(subnormal, exponent - 64, exponent);

    // ln(m) = 2 atanh((m - 1) / (m + 1)), a series in the ratio.
    let ratio = (mantissa - 1.0) / (1.0 + mantissa);
    let ratio2 = ratio * ratio;
    let ratio3 = ratio * ratio2;
    let series = poly(ratio2, LOG_COEFFICIENTS);
    let result = fma(ratio, 2.0, LN2_HI * f64::cast_from(exponent));
    let result = fma(ratio3, series, result);

    let result = select(d == f64::INFINITY, f64::INFINITY, result);
    let result = select(d < 0.0 || d.is_nan(), f64::NAN, result);
    select(d == 0.0, f64::NEG_INFINITY, result)
}

#[cube]
pub(super) fn pow(x: f64, y: f64) -> f64 {
    let y_is_int = is_int(y);
    let y_is_odd = is_odd(y) && y_is_int;

    let d = mul_df(logk(x.abs()), y);
    let result = expk(d);
    let result = select(d.hi > LOG_DBL_MAX, f64::INFINITY, result);
    let sign = select(y_is_int, select(y_is_odd, -1.0f64, 1.0), f64::NAN);
    let result = result * select(x > 0.0, 1.0f64, sign);

    let efx = mul_sign(x.abs() - 1.0, y);
    let infinite_y = select(efx < 0.0, 0.0, select(efx == 0.0, 1.0, f64::INFINITY));
    let result = select(y.is_inf(), infinite_y, result);

    let x_edge = select(sign_bit(y) != (x == 0.0), 0.0, f64::INFINITY);
    let x_edge = mul_sign(x_edge, select(y_is_odd, x, 1.0));
    let result = select(x.is_inf() || x == 0.0, x_edge, result);

    let result = select(x.is_nan() || y.is_nan(), f64::NAN, result);
    select(y == 0.0 || x == 1.0, 1.0, result)
}

/// `ln(d)` as a double-double, for `pow`.
#[cube]
fn logk(d: f64) -> Dd {
    let subnormal = d < DBL_MIN;
    let scaled = select(subnormal, d * TWO_POW_64, d);
    let exponent = ilogb2k(scaled * (1.0 / 0.75));
    let mantissa = ldexp3(scaled, -exponent);
    let exponent = select(subnormal, exponent - 64, exponent);

    let ratio = div_dd(add2_ff(-1.0, mantissa), add2_ff(1.0, mantissa));
    let ratio2 = square(ratio);
    let series = poly(ratio2.hi, LOGK_COEFFICIENTS);

    let sum = mul_df(dd(LN2_HI, LN2_LO), f64::cast_from(exponent));
    let sum = add_dd(sum, scale(ratio, 2.0));
    let power = mul_dd(ratio2, ratio);
    let two_thirds = dd(LOGK_TWO_THIRDS_HI, LOGK_TWO_THIRDS_LO);
    let sum = add_dd(sum, mul_dd(power, two_thirds));
    let power = mul_dd(ratio2, power);
    add_dd(sum, mul_df(power, series))
}

/// `exp(d)` of a double-double, for `pow` and the large hyperbolic functions.
#[cube]
pub(super) fn expk(d: Dd) -> f64 {
    let quotient = ((d.hi + d.lo) * R_LN2).round();
    let exponent = i32::cast_from(quotient);
    let reduced = add2_df(d, quotient * -L2U);
    let reduced = add2_df(reduced, quotient * -L2L);
    let reduced = normalize(reduced);

    let series = poly(reduced.hi, EXPK_COEFFICIENTS);
    let sum = add_fd(1.0, reduced);
    let sum = add_dd(sum, mul_df(square(reduced), series));
    let result = ldexp2(sum.hi + sum.lo, exponent);
    select(d.hi < -1000.0, 0.0, result)
}

/// `ln(d)` of a positive double-double, for the inverse hyperbolic functions.
#[cube]
pub(super) fn logk2(d: Dd) -> Dd {
    let exponent = ilogbk(d.hi * (1.0 / 0.75));
    let mantissa = dd(ldexp2(d.hi, -exponent), ldexp2(d.lo, -exponent));

    let ratio = div_dd(add2_df(mantissa, -1.0), add2_df(mantissa, 1.0));
    let ratio2 = square(ratio);
    let series = poly(ratio2.hi, LOGK2_COEFFICIENTS);
    let series = fma(series, ratio2.hi, LOGK2_TWO_THIRDS);

    let sum = mul_df(dd(LN2_HI, LN2_LO), f64::cast_from(exponent));
    let sum = add_dd(sum, scale(ratio, 2.0));
    add_dd(sum, mul_df(mul_dd(ratio2, ratio), series))
}
