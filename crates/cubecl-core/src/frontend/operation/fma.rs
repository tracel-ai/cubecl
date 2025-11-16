use cubecl_macros::{cube, intrinsic};

use crate::{
    self as cubecl,
    ir::{Arithmetic, FmaOperator, Instruction},
    prelude::CubePrimitive,
};

/// Fused multiply-add `A*B+C`.
#[cube]
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    intrinsic!(|scope| {
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
    })
}
