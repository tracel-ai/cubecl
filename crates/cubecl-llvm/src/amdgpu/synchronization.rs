//! AMDGPU synchronization.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
use pliron_llvm::attributes::AtomicOrderingAttr;
use pliron_llvm::attributes::SyncScopeAttr;
use pliron_llvm::ops as llvm;
use pliron_llvm::types::VoidType;

use crate::shared::intrinsic::call_op;

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
