use cubecl_ir::pliron::value::Value;

use crate::prelude::*;
use crate::{self as cubecl};

define_scalar!(ElemA);
define_size!(SizeA);

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

#[allow(missing_docs)]
pub fn expand_hypot(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    hypot::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
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

#[allow(missing_docs)]
pub fn expand_rhypot(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    rhypot::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}
