use crate::{
    ir::{Arithmetic, ExpandElement, FmaOperator, Instruction, Scope},
    prelude::CubePrimitive,
    unexpanded,
};

/// Fused multiply-add `A*B+C`.
#[allow(unused_variables)]
pub fn fma<C: CubePrimitive>(a: C, b: C, c: C) -> C {
    unexpanded!()
}

/// Expand method of [fma].
#[allow(unused_variables)]
pub fn fma_expand<C: CubePrimitive>(
    scope: &mut Scope,
    a: ExpandElement,
    b: ExpandElement,
    c: ExpandElement,
) -> ExpandElement {
    let output = scope.create_local(a.ty);

    let out = *output;
    let a = *a;
    let b = *b;
    let c = *c;

    scope.register(Instruction::new(
        Arithmetic::Fma(FmaOperator { a, b, c }),
        out,
    ));

    output
}
