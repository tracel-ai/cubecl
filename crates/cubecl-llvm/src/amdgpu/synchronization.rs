//! Lowering of `sync.sync` to the hardware's own barriers.
//!
//! A cube is a workgroup, so a cube barrier is `llvm.amdgcn.s.barrier` between workgroup scoped
//! fences: the fences order the shared memory, the barrier the execution.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
use pliron_llvm::attributes::AtomicOrderingAttr;
use pliron_llvm::ops as llvm;
use pliron_llvm::types::FuncType;
use pliron_llvm::types::VoidType;

/// The workgroup barrier, i.e. what `__syncthreads` compiles to.
const S_BARRIER: &str = "llvm.amdgcn.s.barrier";

/// Emits a fence over the memory a cube shares, in `ordering`.
fn workgroup_fence(scope: &Scope, ordering: AtomicOrderingAttr) {
    let fence = llvm::FenceOp::new(scope.ctx_mut(), ordering, Some("workgroup".into()));
    scope.register(&fence);
}

/// Blocks until every unit of the cube reached this point, and makes what each of them wrote
/// before it visible to all the others.
pub fn lower_sync_cube(scope: &Scope) {
    workgroup_fence(scope, AtomicOrderingAttr::Release);

    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let barrier_ty = FuncType::get(scope.ctx_mut(), void_ty, vec![], false);
    let barrier = llvm::CallIntrinsicOp::new(scope.ctx_mut(), S_BARRIER.into(), barrier_ty, vec![]);
    scope.register(&barrier);

    workgroup_fence(scope, AtomicOrderingAttr::Acquire);
}
