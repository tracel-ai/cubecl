use crate::{prelude::*, unexpanded};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Expand method of [`fma()`].
pub mod fma {
    use super::*;
    use cubecl_ir::{Arithmetic, FmaOperands, Instruction, Scope};

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        a: NativeExpand<C>,
        b: NativeExpand<C>,
        c: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let output = scope.create_local(a.expand.value_type());
        let a = a.expand;
        let b = b.expand;
        let c = c.expand;

        scope.register(Instruction::new(
            Arithmetic::Fma(FmaOperands { a, b, c }),
            output,
        ));

        output.into()
    }
}
