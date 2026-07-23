use cubecl_core::ir::{dialect::vector::*, prelude::*};

use crate::metal::metal_op_with_out;

metal_op_with_out!(MagnitudeOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    format!("length({input})")
});

metal_op_with_out!(NormalizeOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    format!("normalize({input})")
});

metal_op_with_out!(FDotOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("dot({lhs}, {rhs})")
});
