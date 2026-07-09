use cubecl_macros::intrinsic;

use crate as cubecl;
use crate::prelude::{CubePrimitive, Vector};
use crate::{ir::Scope, prelude::*};

/// Executes both branches, *then* selects a value based on the condition. This *should* be
/// branchless, but might depend on the compiler.
///
/// # Safety
///
/// Since both branches are *evaluated* regardless of the condition, both branches must be *valid*
/// regardless of the condition. Illegal memory accesses should not be done in either branch.
pub fn select<C: CubePrimitive>(condition: bool, then: C, or_else: C) -> C {
    if condition { then } else { or_else }
}

/// Same as [`select()`] but with vectors instead.
#[cube]
pub fn select_many<C: Scalar, N: Size>(
    condition: Vector<bool, N>,
    then: Vector<C, N>,
    or_else: Vector<C, N>,
) -> Vector<C, N> {
    intrinsic!(|scope| select::expand(scope, condition.expand.into(), then, or_else))
}

pub mod select {
    use cubecl_ir::{ExpandValue, dialect::general::SelectOp};

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        condition: NativeExpand<bool>,
        then: NativeExpand<C>,
        or_else: NativeExpand<C>,
    ) -> NativeExpand<C> {
        if let ExpandValue::Constant { value, .. } = condition.expand {
            if value.as_bool() {
                return then;
            } else {
                return or_else;
            }
        }

        let condition = condition.read_value(scope);
        let then = then.read_value(scope);
        let or_else = or_else.read_value(scope);

        let [condition, then, or_else] =
            normalize_same_vectorization(scope, [condition, then, or_else]);
        let select = SelectOp::new(scope.ctx_mut(), condition, then, or_else);
        scope.register_with_result(&select).into()
    }
}
