use cubecl_ir::{ExpandElement, Variable};

use crate::prelude::*;
use crate::{self as cubecl};

/// Computes the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
#[cube]
pub fn hypot<F: Float>(lhs: Line<F>, rhs: Line<F>) -> Line<F> {
    let one = Line::empty(lhs.size()).fill(F::from_int(1));
    let a = lhs.abs();
    let b = rhs.abs();
    let max_val = max(a, b);
    let max_val_is_zero = max_val.equal(Line::empty(lhs.size()).fill(F::from_int(0)));
    let max_val_safe = select_many(max_val_is_zero, one, max_val);
    let min_val = min(a, b);
    let t = min_val / max_val_safe;

    max_val * fma(t, t, one).sqrt()
}

#[allow(missing_docs)]
pub fn expand_hypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_type::<FloatExpand<0>>(lhs.ty.storage_type());
    let res = hypot::expand::<FloatExpand<0>>(
        scope,
        ExpandElement::Plain(lhs).into(),
        ExpandElement::Plain(rhs).into(),
    );
    assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
}

/// Computes the reciprocal of the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `1 / sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
#[cube]
pub fn rhypot<F: Float>(lhs: Line<F>, rhs: Line<F>) -> Line<F> {
    let one = Line::empty(lhs.size()).fill(F::from_int(1));
    let a = lhs.abs();
    let b = rhs.abs();
    let max_val = max(a, b);
    let max_val_is_zero = max_val.equal(Line::empty(lhs.size()).fill(F::from_int(0)));
    let max_val_safe = select_many(max_val_is_zero, one, max_val);
    let min_val = min(a, b);
    let t = min_val / max_val_safe;

    fma(t, t, one).inverse_sqrt() / max_val
}

#[allow(missing_docs)]
pub fn expand_rhypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_type::<FloatExpand<0>>(lhs.ty.storage_type());
    let res = rhypot::expand::<FloatExpand<0>>(
        scope,
        ExpandElement::Plain(lhs).into(),
        ExpandElement::Plain(rhs).into(),
    );
    assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
}
