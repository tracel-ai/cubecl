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
    if condition { then } else { or_else }
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

/// Returns the value at `index` in `slice` if `condition` is `true`, otherwise returns `fallback`.
///
/// This function is designed to be branchless while avoiding the read operation when `condition` is `false`.
/// Actual behavior may depend on compiler optimizations.
///
/// # Safety
///
/// Unlike [`select`], no read is performed unless `condition` is `true`.
pub fn conditional_read<C: CubePrimitive, I: Index>(
    _condition: bool,
    _slice: Slice<C>,
    _index: I,
    _fallback: C,
) -> C {
    unexpanded!()
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

pub mod conditional_read {
    use std::num::NonZero;

    use cubecl_ir::ConditionalRead;

    use crate::ir::Instruction;

    use super::*;

    pub fn expand<C: CubePrimitive, I: Index>(
        scope: &mut Scope,
        condition: ExpandElementTyped<bool>,
        slice: ExpandElementTyped<Slice<C>>,
        index: ExpandElementTyped<u32>,
        fallback: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        let cond = condition.expand.consume();
        let slice = slice.expand.consume();
        let index = index.expand.consume();
        let fallback = fallback.expand.consume();

        let vf = cond.vectorization_factor();
        let vf = Ord::max(vf, slice.vectorization_factor());
        let vf = Ord::max(vf, fallback.vectorization_factor());

        let output = scope.create_local(slice.item.vectorize(NonZero::new(vf)));
        let out = *output;

        let conditional_read = Operator::ConditionalRead(ConditionalRead {
            cond,
            slice,
            index,
            fallback,
        });
        scope.register(Instruction::new(conditional_read, out));

        output.into()
    }
}
