use cubecl_core::ir::dialect::plane::*;

use crate::hip::hip_op_with_out;

hip_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("__shfl({val}, {lane});")
});

hip_op_with_out!(ShuffleOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).name(ctx);
    format!("__shfl({val}, {lane});")
});

hip_op_with_out!(ShuffleXorOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let mask = op.mask(ctx).name(ctx);
    format!("__shfl_xor({val}, {mask});")
});

hip_op_with_out!(ShuffleUpOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_up({val}, {delta});")
});

hip_op_with_out!(ShuffleDownOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_down({val}, {delta});")
});

hip_op_with_out!(AllOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("static_cast<bool>(__all({val}));")
});

hip_op_with_out!(AnyOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("static_cast<bool>(__any({val}));")
});

hip_op_with_out!(BallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("{{__ballot({val}), 0, 0, 0}};")
});
