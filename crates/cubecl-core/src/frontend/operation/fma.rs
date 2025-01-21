use crate::{
    ir::{Arithmetic, ExpandElement, FmaOperator, Instruction},
    prelude::{CubeContext, CubePrimitive},
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
    context: &mut CubeContext,
    a: ExpandElement,
    b: ExpandElement,
    c: ExpandElement,
) -> ExpandElement {
    let output = context.create_local(a.item);

    let out = *output;
    let a = *a;
    let b = *b;
    let c = *c;

    context.register(Instruction::new(
        Arithmetic::Fma(FmaOperator { a, b, c }),
        out,
    ));

    output
}
