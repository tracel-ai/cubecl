//! `sin`, `cos` and `tan` to 3.5 ulp, from SLEEF `xsin`, `xcos`, `xtan` and `rempi`.
//!
//! An argument below 15 is reduced by pi in two parts, one below 1e14 by pi in four parts, and a
//! larger one by Payne-Hanek reduction against the bits of 1/(2 pi).

use cubecl_core::{self as cubecl, prelude::*};

use super::base::{FRAC_1_PI, FRAC_2_PI, PI_HI, PI_LO, ilogb2k, ldexp2, ldexp3, mul_sign, poly};
use super::double_double::{Dd, add2_dd, dd, mul_dd, mul_df, mul_ff, normalize};

// pi in four parts for arguments below `TRIG_RANGE_MAX`.
const PI_A: f64 = 3.141_592_621_803_283_7;
const PI_B: f64 = 3.178_650_942_459_171_3e-8;
const PI_C: f64 = 1.224_646_786_410_718_9e-16;
const PI_D: f64 = 1.273_663_432_702_19e-24;
const TRIG_RANGE_MAX: f64 = 1e14;

// pi in two parts for arguments below `TRIG_RANGE_MAX2`. The head has three trailing zero bits,
// so its product with a small integer is exact.
const PI_A2: f64 = PI_HI;
const PI_B2: f64 = PI_LO;
const TRIG_RANGE_MAX2: f64 = 15.0;

/// `tan` uses Payne-Hanek reduction from here on.
const TAN_RANGE_MAX: f64 = 1e6;

/// 2^24, the split of the quotient for the four-part reduction.
const SPLIT: f64 = 16_777_216.0;

const SIN_COEFFICIENTS: [f64; 8] = [
    -7.972_559_550_090_379e-18,
    2.810_099_727_108_632e-15,
    -7.647_122_191_181_588e-13,
    1.605_904_306_056_645e-10,
    -2.505_210_837_635_020_5e-8,
    2.755_731_922_391_987_5e-6,
    -0.000_198_412_698_412_696_16,
    0.008_333_333_333_333_33,
];
const SIN_LAST: f64 = -0.166_666_666_666_666_66;

const TAN_COEFFICIENTS: [f64; 8] = [
    0.000_324_509_882_663_927_63,
    0.000_561_921_973_811_432_4,
    0.001_460_781_502_402_784_5,
    0.003_591_611_540_792_499_5,
    0.008_863_268_409_563_113,
    0.021_869_487_281_855_355,
    0.053_968_253_995_172_73,
    0.133_333_333_333_050_06,
];
const TAN_LAST: f64 = 0.333_333_333_333_334_37;

/// The bits of 1/(2 pi) after the binary point, 32 to a word, most significant first.
const INV_TWO_PI_WORDS: [u32; 38] = [
    0x28be_60db,
    0x9391_054a,
    0x7f09_d5f4,
    0x7d4d_3770,
    0x36d8_a566,
    0x4f10_e410,
    0x7f94_58ea,
    0xf7ae_f158,
    0x6dc9_1b8e,
    0x9093_74b8,
    0x0192_4bba,
    0x8274_6487,
    0x3f87_7ac7,
    0x2c4a_69cf,
    0xba20_8d7d,
    0x4bae_d121,
    0x3a67_1c09,
    0xad17_df90,
    0x4e64_758e,
    0x60d4_ce7d,
    0x2721_17e2,
    0xef7e_4a0e,
    0xc7fe_25ff,
    0xf781_6603,
    0xfbcb_c462,
    0xd682_9b47,
    0xdb4d_9fb3,
    0xc9f2_c26d,
    0xd3d1_8fd9,
    0xa797_fa8b,
    0x5d49_eeb1,
    0xfaf9_7c5e,
    0xcf41_ce7d,
    0xe294_a4ba,
    0x9afe_d7ec,
    0x47e3_5742,
    0x1580_cc11,
    0xbf1e_daea,
];

#[cube]
pub(super) fn sin(x: f64) -> f64 {
    let quotient = (x * FRAC_1_PI).round();
    let mut half_turns = i32::cast_from(quotient);
    let mut reduced = fma(quotient, -PI_A2, x);
    reduced = fma(quotient, -PI_B2, reduced);

    let in_two_part_range = x.abs() < TRIG_RANGE_MAX2;
    if !in_two_part_range {
        let quotient_high = (x * (FRAC_1_PI / SPLIT)).trunc() * SPLIT;
        let quotient_low = fma(x, FRAC_1_PI, -quotient_high).round();
        half_turns = i32::cast_from(quotient_low);
        reduced = reduce_four_parts(x, quotient_high, quotient_low, 1.0);

        let in_four_part_range = x.abs() < TRIG_RANGE_MAX;
        if !in_four_part_range {
            let (remainder, quarter_turns) = rempi(x);
            let parity = select(remainder.hi > 0.0, 2i32, 1);
            half_turns = ((quarter_turns & 3) * 2 + parity) >> 2;
            let half_pi = dd(
                mul_sign(-PI_HI * 0.5, remainder.hi),
                mul_sign(-PI_LO * 0.5, remainder.hi),
            );
            let shifted = add2_dd(remainder, half_pi);
            let odd = (quarter_turns & 1) == 1;
            let value = select(odd, shifted.hi + shifted.lo, remainder.hi + remainder.lo);
            reduced = select(x.is_inf() || x.is_nan(), f64::NAN, value);
        }
    }

    let square = reduced * reduced;
    let reduced = select((half_turns & 1) == 1, -reduced, reduced);
    let series = fma(poly(square, SIN_COEFFICIENTS), square, SIN_LAST);
    let result = square * (series * reduced) + reduced;
    // Keeps the sign of a zero.
    select(x == 0.0, x, result)
}

#[cube]
pub(super) fn cos(x: f64) -> f64 {
    let quotient = fma(2.0, fma(x, FRAC_1_PI, -0.5).round(), 1.0);
    let mut quarter_turns = i32::cast_from(quotient);
    let mut reduced = fma(quotient, -PI_A2 * 0.5, x);
    reduced = fma(quotient, -PI_B2 * 0.5, reduced);

    let in_two_part_range = x.abs() < TRIG_RANGE_MAX2;
    if !in_two_part_range {
        let quotient_high = fma(x, FRAC_1_PI / (SPLIT * 0.5), -FRAC_1_PI / SPLIT).trunc();
        let quotient_low = (x * FRAC_1_PI + fma(quotient_high, -SPLIT * 0.5, -0.5)).round();
        let quotient_low = i32::cast_from(quotient_low) * 2 + 1;
        quarter_turns = quotient_low;
        let quotient_high = quotient_high * SPLIT;
        reduced = reduce_four_parts(x, quotient_high, f64::cast_from(quotient_low), 0.5);

        let in_four_part_range = x.abs() < TRIG_RANGE_MAX;
        if !in_four_part_range {
            let (remainder, rempi_turns) = rempi(x);
            let parity = select(remainder.hi > 0.0, 8i32, 7);
            quarter_turns = ((rempi_turns & 3) * 2 + parity) >> 1;
            let sign = select(remainder.hi > 0.0, 0.0f64, -1.0);
            let half_pi = dd(mul_sign(-PI_HI * 0.5, sign), mul_sign(-PI_LO * 0.5, sign));
            let shifted = add2_dd(remainder, half_pi);
            let even = (rempi_turns & 1) == 0;
            let value = select(even, shifted.hi + shifted.lo, remainder.hi + remainder.lo);
            reduced = select(x.is_inf() || x.is_nan(), f64::NAN, value);
        }
    }

    let square = reduced * reduced;
    let reduced = select((quarter_turns & 2) == 0, -reduced, reduced);
    let series = fma(poly(square, SIN_COEFFICIENTS), square, SIN_LAST);
    square * (series * reduced) + reduced
}

#[cube]
pub(super) fn tan(x: f64) -> f64 {
    let quotient = (x * FRAC_2_PI).round();
    let mut quarter_turns = i32::cast_from(quotient);
    let mut reduced = fma(quotient, -PI_A2 * 0.5, x);
    reduced = fma(quotient, -PI_B2 * 0.5, reduced);

    let in_two_part_range = x.abs() < TRIG_RANGE_MAX2;
    if !in_two_part_range {
        let quotient_high = (x * (FRAC_2_PI / SPLIT)).trunc() * SPLIT;
        let quotient_low = (x * FRAC_2_PI - quotient_high).round();
        quarter_turns = i32::cast_from(quotient_low);
        reduced = reduce_four_parts(x, quotient_high, quotient_low, 0.5);

        let in_four_part_range = x.abs() < TAN_RANGE_MAX;
        if !in_four_part_range {
            let (remainder, rempi_turns) = rempi(x);
            quarter_turns = rempi_turns;
            let value = remainder.hi + remainder.lo;
            reduced = select(x.is_inf() || x.is_nan(), f64::NAN, value);
        }
    }

    // tan(r) = 2 tan(r/2) / (1 - tan(r/2)^2), with tan(r/2) from the series.
    let half = reduced * 0.5;
    let square = half * half;
    let series = fma(poly(square, TAN_COEFFICIENTS), square, TAN_LAST);
    let tan_half = fma(square, series * half, half);

    let denominator = fma(tan_half, tan_half, -1.0);
    let numerator = tan_half * -2.0;
    let odd = (quarter_turns & 1) == 1;
    let result = select(odd, -denominator, numerator) / select(odd, numerator, denominator);
    // Keeps the sign of a zero.
    select(x == 0.0, x, result)
}

/// `r - (dqh + dql) * pi * factor`, with pi in four parts.
#[cube]
fn reduce_four_parts(r: f64, dqh: f64, dql: f64, #[comptime] factor: f64) -> f64 {
    let mut u = fma(dqh, f64::new(comptime![-PI_A * factor]), r);
    u = fma(dql, f64::new(comptime![-PI_A * factor]), u);
    u = fma(dqh, f64::new(comptime![-PI_B * factor]), u);
    u = fma(dql, f64::new(comptime![-PI_B * factor]), u);
    u = fma(dqh, f64::new(comptime![-PI_C * factor]), u);
    u = fma(dql, f64::new(comptime![-PI_C * factor]), u);
    fma(dqh + dql, f64::new(comptime![-PI_D * factor]), u)
}

/// Payne-Hanek reduction: `a` modulo pi/2 as a double-double in [-pi/4, pi/4], and the
/// quadrant.
#[cube]
fn rempi(a: f64) -> (Dd, i32) {
    let mut table = Array::<u32>::new(INV_TWO_PI_WORDS.len());
    #[unroll]
    for i in 0..INV_TWO_PI_WORDS.len() {
        table[i] = comptime![INV_TWO_PI_WORDS[i]];
    }

    let ex = ilogb2k(a) - 55;
    // A very large argument is scaled down by 2^64, and its table row up by as much, so that
    // the products below stay normal.
    let a = ldexp3(a, select(ex > 645, -64i32, 0));
    let ex = select(ex < 0, 0, ex);

    let x = mul_ff(a, rempi_part(table, ex, 0u32));
    let (reduced, q0) = reduce_quarter(x.hi);
    let x = normalize(dd(reduced, x.lo));
    let x = add2_dd(x, mul_ff(a, rempi_part(table, ex, 1u32)));
    let (reduced, q1) = reduce_quarter(x.hi);
    let x = normalize(dd(reduced, x.lo));
    let tail = dd(rempi_part(table, ex, 2u32), rempi_part(table, ex, 3u32));
    let x = normalize(add2_dd(x, mul_df(tail, a)));
    let x = mul_dd(x, dd(PI_HI * 2.0, PI_LO * 2.0));

    let small = a.abs() < 0.7;
    (
        dd(select(small, a, x.hi), select(small, 0.0, x.lo)),
        q0 + q1,
    )
}

/// `x - round(4x) / 4`, and `round(4x) - 4 round(x)`.
#[cube]
fn reduce_quarter(x: f64) -> (f64, i32) {
    let quarters = (x * 4.0).round();
    let q = i32::cast_from(quarters - x.round() * 4.0);
    (x - quarters * 0.25, q)
}

/// One of the four parts of SLEEF's `Sleef_rempitabdp` row for `ex`: `frac(2^k / (2 pi)) / 2^k`
/// with `k = ex + 2`, cut into 52-bit parts. Rows past 645 carry an extra 2^64.
#[cube]
fn rempi_part(table: Array<u32>, ex: i32, #[comptime] part: u32) -> f64 {
    let start = u32::cast_from(ex + 2) + comptime![part * 52];
    let high = bits_after(table, start) >> 12;
    let low = bits_after(table, start + 20);
    // Exact: the sum has 52 bits.
    let chunk = fma(f64::cast_from(high), 4_294_967_296.0, f64::cast_from(low));
    let scale = select(ex > 645, 64i32, 0);
    ldexp2(chunk, scale - i32::cast_from(start) - 52)
}

/// The 32 bits of 1/(2 pi) that follow bit `position`.
#[cube]
fn bits_after(table: Array<u32>, position: u32) -> u32 {
    let index = usize::cast_from(position >> 5);
    let offset = position & 31;
    let high = table[index] << offset;
    let low = table[index + 1] >> (32 - offset);
    select(offset == 0, high, high | low)
}
