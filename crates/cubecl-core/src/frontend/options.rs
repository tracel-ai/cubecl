use cubecl_ir::{FastMath, Scope};
use enumset::EnumSet;

pub fn fast_math_expand<R>(
    scope: &mut Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&mut Scope) -> R,
) -> R {
    let prev = scope.modes.borrow().fp_math_mode;
    scope.modes.borrow_mut().fp_math_mode = value;
    let res = body(scope);
    scope.modes.borrow_mut().fp_math_mode = prev;

    res
}
