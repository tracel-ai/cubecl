use cubecl_core::ir::{dialect::cmp::*, prelude::*};

use crate::{metal::metal_op_with_out, shared::convert::no_msl_bfloat};

metal_op_with_out!(SClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({input}, {min}, {max})")
});

metal_op_with_out!(UClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({input}, {min}, {max})")
});

metal_op_with_out!(FClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({input}, {min}, {max})")
});
no_msl_bfloat!(FClampOp);
