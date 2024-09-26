use crate::prelude::CubePrimitive;

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

pub mod select {
    use crate::{
        ir::{Branch, Select},
        prelude::*,
    };

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        condition: ExpandElementTyped<bool>,
        then: ExpandElementTyped<C>,
        or_else: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        let cond = condition.expand.consume();
        let then = then.expand.consume();
        let or_else = or_else.expand.consume();

        let output = context.create_local_binding(then.item());
        let out = *output;

        context.register(Branch::Select(Select {
            cond,
            then,
            or_else,
            out,
        }));

        output.into()
    }
}
