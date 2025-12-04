use cubecl_ir::{ExpandElement, Variable};

use crate::prelude::*;
use crate::{self as cubecl};

/// Computes the hypotenuse of a right triangle given the lengths of the other two sides.
///
/// This function computes `sqrt(x² + y²)` in a numerically stable way that avoids
/// overflow and underflow issues.
///
/// # Arguments
///
/// * `x` - Length of one side
/// * `y` - Length of the other side
///
/// # Returns
///
/// The length of the hypotenuse
///
/// # Example
///
/// ```rust,ignore
/// let hyp = hypot(F::new(3.0), F::new(4.0));
/// assert!((hyp - F::new(5.0)).abs() < F::new(1e-6));
/// ```
#[cube]
pub fn hypot<F: Float>(lhs: F, rhs: F) -> F {
    let one = F::from_int(1);
    let a = F::abs(lhs);
    let b = F::abs(rhs);
    let max_val = F::max(a, b);
    let max_val_is_zero = max_val == F::from_int(0);
    let max_val_safe = select(max_val_is_zero, one, max_val);
    let min_val = F::min(a, b);
    let t = min_val / max_val_safe;

    max_val * F::sqrt(one + (t * t))
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

#[cube]
pub fn rhypot<F: Float>(lhs: F, rhs: F) -> F {
    let one = F::from_int(1);
    let a = F::abs(lhs);
    let b = F::abs(rhs);
    let max_val = F::max(a, b);
    let max_val_is_zero = max_val == F::from_int(0);
    let max_val_safe = select(max_val_is_zero, one, max_val);
    let min_val = F::min(a, b);
    let t = min_val / max_val_safe;

    F::inverse_sqrt(one + (t * t)) / max_val
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
