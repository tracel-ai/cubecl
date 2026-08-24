use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{leading_part, trailing_part};

const LN2_HI: f32 = leading_part(core::f64::consts::LN_2);
const LN2_LO: f32 = trailing_part(core::f64::consts::LN_2);
const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// What a subnormal is multiplied by to reach the normals, and the exponent that buys.
const SUBNORMAL_SCALE: f32 = (1u32 << 24) as f32;
const SUBNORMAL_SHIFT: i32 = 24;

// Least worst-case relative error fit of `(ln(1+f) - f + f^2/2) / f^3` on
// `[sqrt(1/2) - 1, sqrt(2) - 1]`, the mantissa window the fold leaves, by Remez exchange
// at degree seven. Rounded to `f32` they hold `ln(1+f)` to 25 bits, where an `f32`
// carries 24; the test at the foot of this file is the cross-check.
const LOG_0: f32 = 0.3333333;
const LOG_1: f32 = -0.25000304;
const LOG_2: f32 = 0.20001201;
const LOG_3: f32 = -0.16641581;
const LOG_4: f32 = 0.14209945;
const LOG_5: f32 = -0.12989277;
const LOG_6: f32 = 0.12655699;
const LOG_7: f32 = -0.079742186;

/// `ln x` as the exponent times `ln 2`, plus a series on the mantissa.
///
/// The mantissa is folded at the geometric mean of its octave so the series argument is
/// centred on zero; left in `[1, 2)` the top of the range would need twice the terms. The
/// series is the division-free form, `f - f^2/2 + f^3 P(f)`, which trades five fused
/// multiply-adds for the divide the `atanh` form needs, and folds as a tree rather than a
/// chain, which costs one multiply and halves the depth.
///
/// Evaluated in single precision whatever the argument's own format.
#[cube]
pub fn ln<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let x = Vector::<f32, N>::cast_from(x);

    // A subnormal holds no exponent to extract, so it is scaled into the normals first and
    // the scaling taken back off the exponent afterwards.
    let subnormal = x.less_than(&Vector::new(f32::MIN_POSITIVE));
    let scaled = select_many(subnormal, x * Vector::new(SUBNORMAL_SCALE), x);
    let bits = Vector::<u32, N>::reinterpret(scaled);
    let exponent = Vector::<i32, N>::cast_from(bits >> Vector::new(23u32)) - Vector::new(127i32);
    let mantissa = Vector::<f32, N>::reinterpret(
        (bits & Vector::new(0x007f_ffffu32)) | Vector::new(0x3f80_0000u32),
    );

    let exponent = select_many(subnormal, exponent - Vector::new(SUBNORMAL_SHIFT), exponent);

    let halved = mantissa.greater_than(&Vector::new(SQRT_2));
    let mantissa = select_many(halved, mantissa * Vector::new(0.5f32), mantissa);
    let exponent = select_many(halved, exponent + Vector::new(1i32), exponent);

    // Exact: the mantissa sits in `[1/2, 2]`, where Sterbenz makes the subtraction free of
    // rounding, and the whole accuracy of `ln` near one rests on it.
    let f = mantissa - Vector::new(1.0f32);
    let square = f * f;
    let quartic = square * square;

    let terms_01 = fma(Vector::new(LOG_1), f, Vector::new(LOG_0));
    let terms_23 = fma(Vector::new(LOG_3), f, Vector::new(LOG_2));
    let terms_45 = fma(Vector::new(LOG_5), f, Vector::new(LOG_4));
    let terms_67 = fma(Vector::new(LOG_7), f, Vector::new(LOG_6));

    let low = fma(terms_23, square, terms_01);
    let high = fma(terms_67, square, terms_45);
    let tail = fma(high, quartic, low);

    let mantissa_log = fma(square * f, tail, fma(square, Vector::new(-0.5f32), f));
    let exponent = Vector::<f32, N>::cast_from(exponent);

    let series = fma(
        exponent,
        Vector::new(LN2_HI),
        fma(exponent, Vector::new(LN2_LO), mantissa_log),
    );

    // Nothing above reads the sign bit or asks the exponent field what it means, so every
    // argument that is not a positive finite number arrives here as an ordinary small
    // number rather than as the infinity or the NaN it should be. A zero would otherwise
    // come back near -88, which is a wrong answer that looks like a right one.
    let zero = Vector::<f32, N>::new(0.0);
    let series = select_many(x.greater_than(&zero), series, Vector::new(f32::NAN));
    let series = select_many(x.equal(&zero), Vector::new(f32::NEG_INFINITY), series);
    let series = select_many(
        x.equal(&Vector::new(f32::INFINITY)),
        Vector::new(f32::INFINITY),
        series,
    );

    Vector::<F, N>::cast_from(series)
}

#[cfg(test)]
mod tests {
    use super::super::base::{evaluate, worst_relative_error};
    use super::*;

    /// The coefficients fit `ln(1 + f)` over the mantissa window the fold leaves.
    ///
    /// The fit is measured on the logarithm rather than on the tail polynomial alone,
    /// because the tail carries about a twentieth of the result and an error on it reads
    /// twenty times smaller once reconstructed.
    #[test]
    fn the_series_fits_the_logarithm_over_the_mantissa_window() {
        let from = (0.5f64).sqrt() - 1.0;
        let to = (2.0f64).sqrt() - 1.0;

        let worst = worst_relative_error(from, to, f64::ln_1p, |f| {
            let tail = evaluate(&[LOG_0, LOG_1, LOG_2, LOG_3, LOG_4, LOG_5, LOG_6, LOG_7], f);
            f - f * f / 2.0 + f * f * f * tail
        });

        assert!(worst < 3e-8, "worst relative error {worst}");
    }
}
