use cubecl_macros::intrinsic;

use crate as cubecl;
use crate::prelude::{CubePrimitive, Line};
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

/// Same as [select()] but with lines instead.
#[cube]
#[allow(unused_variables)]
pub fn select_many<C: CubePrimitive>(
    condition: Line<bool>,
    then: Line<C>,
    or_else: Line<C>,
) -> Line<C> {
    intrinsic!(|scope| select::expand(scope, condition.expand.into(), then, or_else))
}

pub mod select {
    use std::num::NonZero;

    use crate::ir::Instruction;

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        condition: ExpandElementTyped<bool>,
        then: ExpandElementTyped<C>,
        or_else: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        let cond = condition.expand.consume();
        let then = then.expand.consume();
        let or_else = or_else.expand.consume();

        let vf = cond.vectorization_factor();
        let vf = Ord::max(vf, then.vectorization_factor());
        let vf = Ord::max(vf, or_else.vectorization_factor());

        let output = scope.create_local(then.item.vectorize(NonZero::new(vf)));
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
