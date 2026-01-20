use crate::{prelude::*, unexpanded};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Expand method of [`fma()`].
pub mod fma {
    use super::*;
    use cubecl_ir::{Arithmetic, FmaOperator, Instruction, Scope};

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        a: ExpandElementTyped<C>,
        b: ExpandElementTyped<C>,
        c: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        let output = scope.create_local(a.expand.ty);
        let out = *output;
        let a = *a.expand;
        let b = *b.expand;
        let c = *c.expand;

        scope.register(Instruction::new(
            Arithmetic::Fma(FmaOperator { a, b, c }),
            out,
        ));

        output.into()
    }
}
