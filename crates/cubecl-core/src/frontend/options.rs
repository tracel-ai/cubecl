use cubecl_ir::EnumSet;
use cubecl_ir::{FastMath, FastMathPermissions, Scope};

pub fn fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    scope.with_fast_math(FastMathPermissions::allowing(value), body)
}

pub fn forbid_fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    scope.with_fast_math(FastMathPermissions::forbidding(value), body)
}
