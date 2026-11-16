use crate::prelude::*;
use crate::{self as cubecl};

/// Computes the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
#[cube]
pub fn hypot<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    let one = Vector::new(F::from_int(1));
    let a = lhs.abs();
    let b = rhs.abs();
    let max_val = max(a, b);
    let max_val_is_zero = max_val.equal(&Vector::new(F::from_int(0)));
    let max_val_safe = select_many(max_val_is_zero, one, max_val);
    let min_val = min(a, b);
    let t = min_val / max_val_safe;

    max_val * fma(t, t, one).sqrt()
}

/// Computes the reciprocal of the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `1 / sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
#[cube]
pub fn rhypot<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    let one = Vector::new(F::from_int(1));
    let a = lhs.abs();
    let b = rhs.abs();
    let max_val = max(a, b);
    let max_val_is_zero = max_val.equal(&Vector::new(F::from_int(0)));
    let max_val_safe = select_many(max_val_is_zero, one, max_val);
    let min_val = min(a, b);
    let t = min_val / max_val_safe;

    fma(t, t, one).inverse_sqrt() / max_val
}

/// Sine and cosine of an angle in degrees. The angle is brought into `[-45, 45]` by whole quarter
/// turns before it becomes radians, a subtraction that is exact while `turns * 90` is, so a
/// multiple of 90 lands on exactly 0 or ±1: `sin(180)` is `0`, where `sin(to_radians(180))` is
/// `-8.7e-8`. A quarter turn is negated as `0 - x`, so `cos(90)` is `+0` and `tan(90)` is `+inf`.
#[cube]
fn sin_cos_degrees<F: Float, N: Size>(input: Vector<F, N>) -> (Vector<F, N>, Vector<F, N>) {
    let zero = Vector::new(F::from_int(0));
    let quarter = Vector::new(F::from_int(90));
    let four = Vector::new(F::from_int(4));
    let turns = (input / quarter).round();
    let radians = (input - turns * quarter).to_radians();
    let (sin, cos) = (radians.sin(), radians.cos());

    let quadrant = turns - (turns / four).floor() * four;
    let one = quadrant.equal(&Vector::new(F::from_int(1)));
    let two = quadrant.equal(&Vector::new(F::from_int(2)));
    let three = quadrant.equal(&Vector::new(F::from_int(3)));
    let odd = one.or(three);

    let sin_out = select_many(odd, cos, sin);
    let cos_out = select_many(odd, sin, cos);
    let sin_out = select_many(two.or(three), zero - sin_out, sin_out);
    let cos_out = select_many(one.or(two), zero - cos_out, cos_out);
    (sin_out, cos_out)
}

/// Computes the sine of an angle given in degrees.
#[cube]
pub fn sind<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    let (sin, _) = sin_cos_degrees(input);
    sin
}

/// Computes the cosine of an angle given in degrees.
#[cube]
pub fn cosd<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    let (_, cos) = sin_cos_degrees(input);
    cos
}

/// Computes the tangent of an angle given in degrees.
#[cube]
pub fn tand<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    let (sin, cos) = sin_cos_degrees(input);
    sin / cos
}

/// Computes the arcsine of a value, in degrees.
#[cube]
pub fn asind<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    input.asin().to_degrees()
}

/// Computes the arccosine of a value, in degrees.
#[cube]
pub fn acosd<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    input.acos().to_degrees()
}

/// Computes the arctangent of a value, in degrees.
#[cube]
pub fn atand<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    input.atan().to_degrees()
}

/// Computes the four-quadrant arctangent of `lhs / rhs`, in degrees.
#[cube]
pub fn atan2d<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    lhs.atan2(rhs).to_degrees()
}
