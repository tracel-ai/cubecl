//! Lowering of `sync.sync` to the hardware's own barriers.
//!
//! A cube is a CTA, so a cube barrier is `barrier.cta.sync.aligned.all` between block-scoped
//! fences: the fences order the shared memory, the barrier the execution. This is what
//! `__syncthreads()` compiles to.
//!
//! A plane is a warp. Its lanes no longer run in lockstep as they did before Volta, so unlike
//! the AMDGPU side's scheduling-only barrier this has to be a real reconvergence:
//! `bar.warp.sync` is what `__syncwarp()` compiles to.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
use pliron_llvm::attributes::AtomicOrderingAttr;
use pliron_llvm::attributes::SyncScopeAttr;
use pliron_llvm::ops as llvm;
use pliron_llvm::types::VoidType;

use crate::shared::intrinsic::{call_op, i32_const_op};

/// The CTA barrier, i.e. what `__syncthreads` compiles to. `aligned` promises every unit of
/// the cube reaches this same barrier, which holds because a cube barrier is a whole-cube
/// operation by definition.
const BARRIER_CTA: &str = "llvm.nvvm.barrier.cta.sync.aligned.all";

/// The warp barrier, i.e. what `__syncwarp` compiles to.
const BARRIER_WARP: &str = "llvm.nvvm.bar.warp.sync";

/// Barrier resource 0, which is the one `__syncthreads` uses and the only one needed while
/// nothing here splits a cube into independently synchronizing groups.
const BARRIER_ID: i32 = 0;

/// Every lane of the warp takes part; see the same constant in [`plane`](super::plane).
const FULL_MASK: i32 = -1;

/// Emits a fence over `sync_scope`, in `ordering`.
fn fence(scope: &Scope, sync_scope: &str, ordering: AtomicOrderingAttr) {
    let fence = llvm::FenceOp::new(
        scope.ctx_mut(),
        ordering,
        SyncScopeAttr::NamedScope(sync_scope.into()),
    );
    scope.register(&fence);
}

/// Emits a call to the valueless intrinsic `name`, over one `i32` operand.
fn barrier(scope: &Scope, name: &str, operand: i32) {
    let void_ty = VoidType::get(scope.ctx_mut()).into();
    let operand = i32_const_op(scope.ctx_mut(), operand);
    scope.register(&operand);
    let arg = operand.get_result(scope.ctx());
    let op = call_op(scope.ctx_mut(), name, void_ty, vec![arg]);
    scope.register(&op);
}

/// Makes what the other lanes of the warp wrote visible here.
///
/// Post-Volta the lanes of a warp can diverge and stay diverged, so this reconverges them
/// rather than only ordering the compiler's scheduling.
pub fn lower_sync_plane(scope: &Scope) {
    fence(scope, "block", AtomicOrderingAttr::AcqRel);
    barrier(scope, BARRIER_WARP, FULL_MASK);
}

/// Blocks until every unit of the cube reached this point, and makes what each of them wrote
/// before it visible to all the others.
pub fn lower_sync_cube(scope: &Scope) {
    fence(scope, "block", AtomicOrderingAttr::Release);
    barrier(scope, BARRIER_CTA, BARRIER_ID);
    fence(scope, "block", AtomicOrderingAttr::Acquire);
}
