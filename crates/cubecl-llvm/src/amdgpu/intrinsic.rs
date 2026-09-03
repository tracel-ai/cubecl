//! The lane index the `llvm.amdgcn` cross-lane lowerings build on.
//!
//! Building the call itself is [`shared::intrinsic`](crate::shared::intrinsic); what is here
//! is the one piece of it AMD does differently. Like everything there, this hands the ops back
//! without inserting them, because the callers insert differently: [`builtins`](super::builtins)
//! appends to a [`Scope`](cubecl_core::ir::Scope), while the [`plane`](super::plane) and
//! [`matrix`](super::matrix) lowerings insert before the op they replace.

use crate::shared::intrinsic::{call_op, i32_const_op, i32_ty};
use crate::shared::to_llvm::prelude::*;

/// Counts the set bits of the exec mask below this lane, which under a full mask is the
/// lane's own index in the wavefront. Split in two halves so the same pair serves both
/// wave widths: on wave32 the high half adds nothing.
const MBCNT_LO: &str = "llvm.amdgcn.mbcnt.lo";
const MBCNT_HI: &str = "llvm.amdgcn.mbcnt.hi";

/// This lane's index within its wavefront: the operations that compute it, in the order they
/// must be inserted, and the value they produce.
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
