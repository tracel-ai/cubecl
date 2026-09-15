use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{leading_part, trailing_part};

const LOG2_E: f32 = core::f32::consts::LOG2_E;

// Split ln(2) for accurate range reduction.
const LN2_HI: f32 = leading_part(core::f64::consts::LN_2);
const LN2_LO: f32 = trailing_part(core::f64::consts::LN_2);

// Degree-six Remez fit of exp(r) over the reduced interval.
const EXP_0: f32 = 1.0;
const EXP_1: f32 = 1.0;
const EXP_2: f32 = 0.4999999;
const EXP_3: f32 = 0.1666642;
const EXP_4: f32 = 0.041668225;
const EXP_5: f32 = 0.008374816;
const EXP_6: f32 = 0.0013836846;

// Input bounds for overflow and underflow.
const EXP_MAX: f32 = 88.72284;
const EXP_MIN: f32 = -104.66522;

#[cube]
pub fn exp<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let x = Vector::<f32, N>::cast_from(x).clamp(Vector::new(EXP_MIN), Vector::new(EXP_MAX));

    let k = (x * Vector::new(LOG2_E)).round();

    let finite = k.equal(&k);
    let k = select_many(finite, k, Vector::new(0.0f32));

    let r = fma(-k, Vector::new(LN2_HI), x);
    let r = fma(-k, Vector::new(LN2_LO), r);

    let square = r * r;
    let quartic = square * square;

    let terms_01 = fma(Vector::new(EXP_1), r, Vector::new(EXP_0));
    let terms_23 = fma(Vector::new(EXP_3), r, Vector::new(EXP_2));
    let terms_45 = fma(Vector::new(EXP_5), r, Vector::new(EXP_4));
    let low = fma(terms_23, square, terms_01);
    let high = fma(Vector::new(EXP_6), square, terms_45);
    let series = fma(high, quartic, low);

    let exponent = Vector::<i32, N>::cast_from(k);
    let half = exponent >> Vector::new(1i32);

    Vector::<F, N>::cast_from(series * power_of_two(half) * power_of_two(exponent - half))
}

#[cube]
fn power_of_two<N: Size>(exponent: Vector<i32, N>) -> Vector<f32, N> {
    Vector::<f32, N>::reinterpret(
        Vector::<u32, N>::cast_from(exponent + Vector::new(127i32)) << Vector::new(23u32),
    )
}

#[cfg(test)]
mod tests {
    use super::super::base::{evaluate, worst_relative_error};
    use super::*;

    #[test]
    fn the_series_fits_the_exponential_over_the_reduced_interval() {
        let half = core::f64::consts::LN_2 / 2.0;
        let worst = worst_relative_error(-half, half, f64::exp, |r| {
            evaluate(&[EXP_0, EXP_1, EXP_2, EXP_3, EXP_4, EXP_5, EXP_6], r)
        });

        assert!(worst < 2e-8, "worst relative error {worst}");
    }
}
