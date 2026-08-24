use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::horner;
use super::exponential::exp;

// Least worst-case relative error fit of `tanh(x)/x` in `x^2` on `[0, 1/4]`, the window
// below where the exponential form stops cancelling, by Remez exchange at degree four.
// Rounded to `f32` they hold `tanh` to 25 bits, where an `f32` carries 24; the test at the
// foot of this file is the cross-check.
const TANH_0: f32 = 1.0;
const TANH_1: f32 = -0.3333307;
const TANH_2: f32 = 0.1332478;
const TANH_3: f32 = -0.052986074;
const TANH_4: f32 = 0.017135007;

/// Where the series and the exponential form change places, which is where `1 - 2/(e+1)`
/// stops losing digits to cancellation.
const SERIES_LIMIT: f32 = 0.5;

/// `tanh x` as a series around zero and as `1 - 2/(e^2x + 1)` away from it.
///
/// Both arms are evaluated on every lane, since a vector has no cheaper way to take one
/// branch, and the exponential's own clamp is what makes the far tail return exactly one.
#[cube]
pub fn tanh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let x = Vector::<f32, N>::cast_from(x);

    let square = x * x;
    let series = x * horner(square, comptime![[TANH_0, TANH_1, TANH_2, TANH_3, TANH_4]]);

    let magnitude = x.abs();
    let doubled = exp(magnitude + magnitude);
    let saturating = Vector::new(1.0f32) - Vector::new(2.0f32) / (doubled + Vector::new(1.0f32));
    let saturating = select_many(x.less_than(&Vector::new(0.0f32)), -saturating, saturating);

    Vector::<F, N>::cast_from(select_many(
        magnitude.less_than(&Vector::new(SERIES_LIMIT)),
        series,
        saturating,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::base::{evaluate, worst_relative_error};
    use super::*;

    /// The series fits `tanh` over the window it is used on, which is where the
    /// exponential form loses digits to cancellation.
    #[test]
    fn the_series_fits_the_tangent_around_zero() {
        let limit = SERIES_LIMIT as f64;
        let worst = worst_relative_error(-limit, limit, f64::tanh, |x| {
            x * evaluate(&[TANH_0, TANH_1, TANH_2, TANH_3, TANH_4], x * x)
        });

        assert!(worst < 4e-8, "worst relative error {worst}");
    }
}
