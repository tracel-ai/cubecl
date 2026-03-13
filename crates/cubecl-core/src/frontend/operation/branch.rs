use cubecl_macros::intrinsic;

use crate as cubecl;
use crate::prelude::{CubePrimitive, Vector};
use crate::{
    ir::{Operator, Scope, Select},
    prelude::*,
};

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
#[allow(unused_variables)]
pub fn select_many<C: Scalar, N: Size>(
    condition: Vector<bool, N>,
    then: Vector<C, N>,
    or_else: Vector<C, N>,
) -> Vector<C, N> {
    intrinsic!(|scope| select::expand(scope, condition.expand.into(), then, or_else))
}

pub mod select {
    use crate::ir::Instruction;

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        condition: NativeExpand<bool>,
        then: NativeExpand<C>,
        or_else: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let cond = condition.expand.consume();
        let then = then.expand.consume();
        let or_else = or_else.expand.consume();

        let vf = cond.vector_size();
        let vf = Ord::max(vf, then.vector_size());
        let vf = Ord::max(vf, or_else.vector_size());

        let output = scope.create_local(then.ty.with_vector_size(vf));
        let out = *output;

        let select = Operator::Select(Select {
            cond,
            then,
            or_else,
        });
        scope.register(Instruction::new(select, out));

        output.into()
    }
}
