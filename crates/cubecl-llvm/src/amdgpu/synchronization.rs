//! AMDGPU synchronization.

use crate::prelude::*;
use pliron_llvm::metadata::{MdNodeAttr, MdOperandAttr, attach_new_metadata};

const S_BARRIER: &str = "llvm.amdgcn.s.barrier";

const WAVE_BARRIER: &str = "llvm.amdgcn.wave.barrier";

const SCHED_BARRIER: &str = "llvm.amdgcn.sched.barrier";

/// The scheduling barrier with mask `0`: the backend's schedulers move no instruction across it.
pub fn lower_schedule_barrier(scope: &Scope) {
    let nothing_crosses = i32_const_op(scope.ctx_mut(), 0);
    scope.register(&nothing_crosses);
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let mask = nothing_crosses.get_result(scope.ctx());
    let op = call_op(scope.ctx_mut(), SCHED_BARRIER, void_ty, vec![mask]);
    scope.register(&op);
}

fn fence(scope: &Scope, sync_scope: &str, ordering: AtomicOrderingAttr) -> llvm::FenceOp {
    fence_at(
        scope,
        SyncScopeAttr::NamedScope(sync_scope.into()),
        ordering,
    )
}

fn fence_at(
    scope: &Scope,
    sync_scope: SyncScopeAttr,
    ordering: AtomicOrderingAttr,
) -> llvm::FenceOp {
    let fence = llvm::FenceOp::new(scope.ctx_mut(), ordering, sync_scope);
    scope.register(&fence);
    fence
}

/// A fence on shared memory alone, which neither waits for the accesses to global memory nor
/// touches their caches.
fn shared_fence(scope: &Scope, sync_scope: &str, ordering: AtomicOrderingAttr) {
    let fence = fence(scope, sync_scope, ordering);
    let ctx = scope.ctx_mut();
    let synchronize_as_local = MdNodeAttr::new_tuple(
        ["amdgpu-synchronize-as", "local"]
            .map(|word| MdOperandAttr::String(word.into()))
            .to_vec(),
    );
    attach_new_metadata(ctx, fence.get_operation(), "mmra", synchronize_as_local)
        .expect("a fence should be inside a module");
}

fn barrier(scope: &Scope, name: &str) {
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let op = call_op(scope.ctx_mut(), name, void_ty, vec![]);
    scope.register(&op);
}

/// Wavefront synchronization requires memory ordering and a scheduling barrier.
pub fn lower_sync_plane(scope: &Scope) {
    shared_fence(scope, "wavefront", AtomicOrderingAttr::AcqRel);
    barrier(scope, WAVE_BARRIER);
}

/// [`lower_sync_plane`] fenced over every address space.
pub fn lower_sync_plane_storage(scope: &Scope) {
    fence(scope, "wavefront", AtomicOrderingAttr::AcqRel);
    barrier(scope, WAVE_BARRIER);
}

/// Cube synchronization requires memory ordering and a workgroup barrier.
pub fn lower_sync_cube(scope: &Scope) {
    shared_fence(scope, "workgroup", AtomicOrderingAttr::Release);
    barrier(scope, S_BARRIER);
    shared_fence(scope, "workgroup", AtomicOrderingAttr::Acquire);
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

/// [`lower_sync_storage`] fenced at the system's scope, LLVM's unnamed one: the release writes
/// back what the caches hold for the host and peer devices to read, and the acquire invalidates
/// what they may have written behind them.
pub fn lower_sync_system(scope: &Scope) {
    fence_at(scope, SyncScopeAttr::System, AtomicOrderingAttr::AcqRel);
    barrier(scope, S_BARRIER);
    fence_at(scope, SyncScopeAttr::System, AtomicOrderingAttr::AcqRel);
}
