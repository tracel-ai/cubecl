use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{horner, leading_part, trailing_part};

// `pi / 2` in three pieces. What the three leave behind is `2^-49`, which is the floor on
// how small a reduced angle can be and still carry its own significance.
const PI_2: f64 = core::f64::consts::FRAC_PI_2;
const PI_2_A: f32 = leading_part(PI_2);
const PI_2_B: f32 = leading_part(PI_2 - PI_2_A as f64);
const PI_2_C: f32 = trailing_part(PI_2 - PI_2_A as f64);

const FRAC_2_PI: f32 = core::f32::consts::FRAC_2_PI;

// Least worst-case relative error fit of `sin(x)/x` in `x^2` on `[0, (pi/4)^2]`, an
// eighth of a turn, by Remez exchange at degree three. Rounded to `f32` they hold `sin`
// to 27 bits, where an `f32` carries 24; the test at the foot of this file is the
// cross-check.
const SIN_0: f32 = 1.0;
const SIN_1: f32 = -0.16666651;
const SIN_2: f32 = 0.008332017;
const SIN_3: f32 = -1.9501822e-04;

// The same fit for `cos(x)` in `x^2` over the same eighth, at degree four, holding `cos`
// to 27 bits.
const COS_0: f32 = 1.0;
const COS_1: f32 = -0.5;
const COS_2: f32 = 0.041666612;
const COS_3: f32 = -0.001388653;
const COS_4: f32 = 2.4372679e-05;

/// The sine of an angle in radians.
///
/// Both series are evaluated and one is discarded, since the reduction they share is the
/// expensive half and nothing downstream keeps the other.
#[cube]
pub fn sin<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let (_, sine) = cos_sin_radians(Vector::<f32, N>::cast_from(x));
    Vector::<F, N>::cast_from(sine)
}

/// The cosine of an angle in radians.
#[cube]
pub fn cos<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let (cosine, _) = cos_sin_radians(Vector::<f32, N>::cast_from(x));
    Vector::<F, N>::cast_from(cosine)
}

/// Cosine and sine of an angle in radians, by reducing onto the eighth of a turn around
/// zero and undoing the quadrant afterwards.
///
/// The reduction subtracts the quadrant's worth of `pi / 2` in three pieces. A single
/// piece would leave an error proportional to the angle, which is the error that makes a
/// hardware `sin` useless far from the origin.
#[cube]
fn cos_sin_radians<N: Size>(x: Vector<f32, N>) -> (Vector<f32, N>, Vector<f32, N>) {
    let quadrant = (x * Vector::new(FRAC_2_PI)).round();

    let offset = fma(-quadrant, Vector::new(PI_2_A), x);
    let offset = fma(-quadrant, Vector::new(PI_2_B), offset);
    let offset = fma(-quadrant, Vector::new(PI_2_C), offset);

    cos_sin_quadrant(offset, Vector::<i32, N>::cast_from(quadrant))
}

/// Cosine and sine of `quadrant` quarter turns plus `offset` radians, for an `offset` no
/// larger than an eighth of a turn.
#[cube]
fn cos_sin_quadrant<N: Size>(
    offset: Vector<f32, N>,
    quadrant: Vector<i32, N>,
) -> (Vector<f32, N>, Vector<f32, N>) {
    let square = offset * offset;

    let sine = offset * horner(square, comptime![[SIN_0, SIN_1, SIN_2, SIN_3]]);
    let cosine = horner(square, comptime![[COS_0, COS_1, COS_2, COS_3, COS_4]]);

    let quadrant = quadrant & Vector::new(3i32);
    let swapped = (quadrant & Vector::new(1i32)).equal(&Vector::new(1i32));
    let cosine_magnitude = select_many(swapped, sine, cosine);
    let sine_magnitude = select_many(swapped, cosine, sine);

    let cosine_negative =
        ((quadrant + Vector::new(1i32)) & Vector::new(2i32)).equal(&Vector::new(2i32));
    let sine_negative = (quadrant & Vector::new(2i32)).equal(&Vector::new(2i32));

    (
        select_many(cosine_negative, -cosine_magnitude, cosine_magnitude),
        select_many(sine_negative, -sine_magnitude, sine_magnitude),
    )
}

#[cfg(test)]
mod tests {
    use super::super::base::{evaluate, worst_relative_error};
    use super::*;

    /// Both series fit their function over the eighth of a turn the reduction leaves.
    #[test]
    fn the_series_fit_over_an_eighth_of_a_turn() {
        let limit = core::f64::consts::FRAC_PI_4;

        let sine = worst_relative_error(-limit, limit, f64::sin, |x| {
            x * evaluate(&[SIN_0, SIN_1, SIN_2, SIN_3], x * x)
        });
        let cosine = worst_relative_error(-limit, limit, f64::cos, |x| {
            evaluate(&[COS_0, COS_1, COS_2, COS_3, COS_4], x * x)
        });

        assert!(sine < 8e-9, "sine worst relative error {sine}");
        assert!(cosine < 6e-9, "cosine worst relative error {cosine}");
    }
}
