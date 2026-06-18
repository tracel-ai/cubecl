use cubecl_core::ir::{dialect::cmp::ClampOp, prelude::*};

use crate::metal::metal_op_with_out;

metal_op_with_out!(ClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({input}, {min}, {max})")
});
