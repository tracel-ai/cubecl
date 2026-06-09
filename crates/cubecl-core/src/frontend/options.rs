use cubecl_ir::{FastMath, Scope};
use enumset::EnumSet;

pub fn fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    let prev = scope.state().modes.fp_math_mode;
    scope.state_mut().modes.fp_math_mode = value;
    let res = body(scope);
    scope.state_mut().modes.fp_math_mode = prev;

    res
}
