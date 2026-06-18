use cubecl_core::ir::dialect::plane::*;

use crate::metal::metal_op_with_out;

metal_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("simd_shuffle({val}, {lane});")
});

metal_op_with_out!(ShuffleOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).name(ctx);
    format!("simd_shuffle({val}, {lane});")
});

metal_op_with_out!(ShuffleXorOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let mask = op.mask(ctx).name(ctx);
    format!("simd_shuffle_xor({val}, {mask});")
});

metal_op_with_out!(ShuffleUpOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("simd_shuffle_up({val}, {delta});")
});

metal_op_with_out!(ShuffleDownOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("simd_shuffle_down({val}, {delta});")
});

metal_op_with_out!(AllOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_all({val});")
});

metal_op_with_out!(AnyOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_any({val});")
});

metal_op_with_out!(BallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("{{simd_ballot({val}), 0, 0}};")
});

metal_op_with_out!(SumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_sum({val});")
});

metal_op_with_out!(ProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_product({val});")
});

metal_op_with_out!(MinOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_min({val});")
});

metal_op_with_out!(MaxOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_max({val});")
});

metal_op_with_out!(InclusiveSumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_sum({val});")
});

metal_op_with_out!(InclusiveProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_product({val});")
});

metal_op_with_out!(ExclusiveSumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_sum({val});")
});

metal_op_with_out!(ExclusiveProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_product({val});")
});
