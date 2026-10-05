//! NVPTX synchronization. CTA and warp barriers also order memory.

use crate::prelude::*;

const BARRIER_CTA: &str = "llvm.nvvm.barrier.cta.sync.aligned.all";

const BARRIER_WARP: &str = "llvm.nvvm.bar.warp.sync";

/// Barrier resource reserved for cube synchronization.
const BARRIER_ID: i32 = 0;

/// All warp lanes participate.
const FULL_MASK: i32 = -1;

fn barrier(scope: &Scope, name: &str, operand: i32) {
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let operand = i32_const_op(scope.ctx_mut(), operand);
    scope.register(&operand);
    let arg = operand.get_result(scope.ctx());
    let op = call_op(scope.ctx_mut(), name, void_ty, vec![arg]);
    scope.register(&op);
}

pub fn lower_sync_plane(scope: &Scope) {
    barrier(scope, BARRIER_WARP, FULL_MASK);
}

pub fn lower_sync_cube(scope: &Scope) {
    barrier(scope, BARRIER_CTA, BARRIER_ID);
}

fn fence(scope: &Scope, ordering: AtomicOrderingAttr) {
    let fence = llvm::FenceOp::new(
        scope.ctx_mut(),
        ordering,
        SyncScopeAttr::NamedScope("device".into()),
    );
    scope.register(&fence);
}

/// A cube barrier fenced at the device's scope on both sides, where CUDA's C++ writes
/// `__threadfence(); __syncthreads();`: the barrier orders memory within the cube only. The fence
/// ahead of it publishes each unit's writes to every cube and orders an earlier load of what
/// another cube published ahead of the barrier. The one after it acquires for each unit's later
/// reads, and releases the whole cube's writes, which the barrier gathered, to whatever the unit
/// then publishes: that release is what a relaxed atomic after the barrier hands on.
pub fn lower_sync_storage(scope: &Scope) {
    fence(scope, AtomicOrderingAttr::AcqRel);
    barrier(scope, BARRIER_CTA, BARRIER_ID);
    fence(scope, AtomicOrderingAttr::AcqRel);
}
