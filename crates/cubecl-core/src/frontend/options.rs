use cubecl_ir::{FastMath, Marker, Scope};
use enumset::EnumSet;

pub fn fast_math_expand<R>(
    scope: &mut Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&mut Scope) -> R,
) -> R {
    let prev = scope.modes.borrow().math_mode;
    scope.modes.borrow_mut().math_mode = value;
    scope.register(Marker::SetFastMath(value));
    let res = body(scope);
    scope.modes.borrow_mut().math_mode = prev;
    scope.register(Marker::SetFastMath(prev));

    res
}
