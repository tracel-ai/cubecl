use crate::{
    ir::{Operator, Scope, Select},
    prelude::*,
};
use crate::{
    prelude::{CubePrimitive, Line},
    unexpanded,
};

/// Executes both branches, *then* selects a value based on the condition. This *should* be
/// branchless, but might depend on the compiler.
///
/// # Safety
///
/// Since both branches are *evaluated* regardless of the condition, both branches must be *valid*
/// regardless of the condition. Illegal memory accesses should not be done in either branch.
pub fn select<C: CubePrimitive>(condition: bool, then: C, or_else: C) -> C {
    if condition {
        then
    } else {
        or_else
    }
}

/// Same as [select()] but with lines instead.
#[allow(unused_variables)]
pub fn select_many<C: CubePrimitive>(
    condition: Line<bool>,
    then: Line<C>,
    or_else: Line<C>,
) -> Line<C> {
    unexpanded!()
}

pub fn conditional_expr<C: CubePrimitive>(condition: bool, then: C, or_else: C) -> C {
    if condition {
        then
    } else {
        or_else
    }
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

pub mod select_many {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        condition: ExpandElementTyped<Line<bool>>,
        then: ExpandElementTyped<Line<C>>,
        or_else: ExpandElementTyped<Line<C>>,
    ) -> ExpandElementTyped<Line<C>> {
        select::expand(scope, condition.expand.into(), then, or_else)
    }
}

pub mod conditional_expr {
    use std::num::NonZero;

    use cubecl_ir::ConditionalExpr;

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

        let cond_expr = Operator::ConditionalExpr(ConditionalExpr {
            cond,
            then,
            or_else,
        });
        scope.register(Instruction::new(cond_expr, out));

        output.into()
    }
}
