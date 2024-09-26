use crate::prelude::CubePrimitive;

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
