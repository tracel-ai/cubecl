//! Lowering of `sync.sync` to the hardware's own barriers.
//!
//! A cube is a workgroup, so a cube barrier is `llvm.amdgcn.s.barrier` between workgroup scoped
//! fences: the fences order the shared memory, the barrier the execution. A plane is a
//! wavefront, whose lanes already run in lockstep, so there the barrier is only a scheduling
//! one and the fence carries the whole meaning.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
use pliron_llvm::attributes::AtomicOrderingAttr;
use pliron_llvm::attributes::SyncScopeAttr;
use pliron_llvm::ops as llvm;
use pliron_llvm::types::VoidType;

use crate::amdgpu::intrinsic::call_op;

/// The workgroup barrier, i.e. what `__syncthreads` compiles to.
const S_BARRIER: &str = "llvm.amdgcn.s.barrier";

/// The wavefront barrier, which orders the compiler's scheduling rather than the hardware.
const WAVE_BARRIER: &str = "llvm.amdgcn.wave.barrier";

/// Emits a fence over `sync_scope`, in `ordering`.
fn fence(scope: &Scope, sync_scope: &str, ordering: AtomicOrderingAttr) {
    let fence = llvm::FenceOp::new(
        scope.ctx_mut(),
        ordering,
        SyncScopeAttr::NamedScope(sync_scope.into()),
    );
    scope.register(&fence);
}

/// Emits a call to the valueless intrinsic `name`.
fn barrier(scope: &Scope, name: &str) {
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let op = call_op(scope.ctx_mut(), name, void_ty, vec![]);
    scope.register(&op);
}

/// Makes what the other lanes of the wavefront wrote visible here.
///
/// The lanes run in lockstep, so there is no execution to synchronize and the barrier only
/// stops the compiler reordering across this point.
pub fn lower_sync_plane(scope: &Scope) {
    fence(scope, "wavefront", AtomicOrderingAttr::AcqRel);
    barrier(scope, WAVE_BARRIER);
}

/// Blocks until every unit of the cube reached this point, and makes what each of them wrote
/// before it visible to all the others.
pub fn lower_sync_cube(scope: &Scope) {
    fence(scope, "workgroup", AtomicOrderingAttr::Release);
    barrier(scope, S_BARRIER);
    fence(scope, "workgroup", AtomicOrderingAttr::Acquire);
}
