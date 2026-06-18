use cubecl_core::ir::dialect::{bitwise::*, math::TanhOp};

use crate::metal::metal_op_with_out;

metal_op_with_out!(TanhOp, |op, ctx| {
    format!("safe_tanh_scalar({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(CountOnesOp, |op, ctx| {
    format!("popcount({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(ReverseBitsOp, |op, ctx| {
    format!("reverse_bits({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(LeadingZerosBitsOp, |op, ctx| {
    format!("clz({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(TrailingZerosBitsOp, |op, ctx| {
    format!("ctz({})", op.input(ctx).name(ctx))
});
