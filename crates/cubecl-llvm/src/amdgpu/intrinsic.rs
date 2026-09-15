//! AMDGPU lane intrinsics.

use crate::shared::intrinsic::{call_op, i32_const_op, i32_ty};
use crate::shared::to_llvm::prelude::*;

const MBCNT_LO: &str = "llvm.amdgcn.mbcnt.lo";
const MBCNT_HI: &str = "llvm.amdgcn.mbcnt.hi";

pub fn lane_id_ops(ctx: &mut Context) -> (Vec<Ptr<Operation>>, Value) {
    let ty = i32_ty(ctx);
    let all_lanes = i32_const_op(ctx, -1);
    let zero = i32_const_op(ctx, 0);

    let lo = call_op(
        ctx,
        MBCNT_LO,
        ty,
        vec![all_lanes.get_result(ctx), zero.get_result(ctx)],
    );
    let hi = call_op(
        ctx,
        MBCNT_HI,
        ty,
        vec![all_lanes.get_result(ctx), lo.get_result(ctx)],
    );

    let lane = hi.get_result(ctx);
    let ops = vec![
        all_lanes.get_operation(),
        zero.get_operation(),
        lo.get_operation(),
        hi.get_operation(),
    ];
    (ops, lane)
}
