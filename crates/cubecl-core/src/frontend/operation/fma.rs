use crate::{prelude::*, unexpanded};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Expand method of [`fma()`].
pub mod fma {
    use super::*;
    use cubecl_ir::{
        Scope, dialect::math::FmaOp, pliron::builtin::op_interfaces::OneResultInterface,
    };

    pub fn expand<C: CubePrimitive>(
        scope: &Scope,
        a: NativeExpand<C>,
        b: NativeExpand<C>,
        c: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let a = a.read_value(scope);
        let b = b.read_value(scope);
        let c = c.read_value(scope);

        let [a, b, c] = normalize_same_vectorization(scope, [a, b, c]);

        let op = FmaOp::new(&mut scope.ctx_mut(), a, b, c);
        scope.register(&op);
        op.get_result(&scope.ctx()).into()
    }
}
