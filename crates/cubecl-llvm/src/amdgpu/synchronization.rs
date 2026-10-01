//! AMDGPU synchronization.

use crate::prelude::*;

const S_BARRIER: &str = "llvm.amdgcn.s.barrier";

const WAVE_BARRIER: &str = "llvm.amdgcn.wave.barrier";

fn fence(scope: &Scope, sync_scope: &str, ordering: AtomicOrderingAttr) {
    let fence = llvm::FenceOp::new(
        scope.ctx_mut(),
        ordering,
        SyncScopeAttr::NamedScope(sync_scope.into()),
    );
    scope.register(&fence);
}

fn barrier(scope: &Scope, name: &str) {
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let op = call_op(scope.ctx_mut(), name, void_ty, vec![]);
    scope.register(&op);
}

/// Wavefront synchronization requires memory ordering and a scheduling barrier.
pub fn lower_sync_plane(scope: &Scope) {
    fence(scope, "wavefront", AtomicOrderingAttr::AcqRel);
    barrier(scope, WAVE_BARRIER);
}

/// Cube synchronization requires memory ordering and a workgroup barrier.
pub fn lower_sync_cube(scope: &Scope) {
    fence(scope, "workgroup", AtomicOrderingAttr::Release);
    barrier(scope, S_BARRIER);
    fence(scope, "workgroup", AtomicOrderingAttr::Acquire);
}

/// A workgroup barrier fenced at the agent's scope on both sides: the fence ahead of it publishes
/// each unit's writes to every cube and orders an earlier load of what another cube published
/// ahead of the barrier. The one after it invalidates the caches a later read would otherwise be
/// served a stale copy from, and releases the whole workgroup's writes, which the barrier
/// gathered, to whatever the unit then publishes through a relaxed atomic.
pub fn lower_sync_storage(scope: &Scope) {
    fence(scope, "agent", AtomicOrderingAttr::AcqRel);
    barrier(scope, S_BARRIER);
    fence(scope, "agent", AtomicOrderingAttr::AcqRel);
}
