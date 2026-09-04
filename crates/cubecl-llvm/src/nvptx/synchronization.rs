//! Lowering of `sync.sync` to the hardware's own barriers.
//!
//! A cube is a CTA, so a cube barrier is `barrier.cta.sync.aligned.all` on its own -- which is
//! exactly what `__syncthreads()` compiles to. No fence goes with it: PTX gives `barrier.cta.sync`
//! the memory ordering as well as the execution ordering, so a `fence.acq_rel.cta` around it is a
//! `membar.cta` the hardware has already performed. It is not free either, and a matmul that
//! synchronizes once per stage pays for it once per stage.
//!
//! A plane is a warp. Its lanes no longer run in lockstep as they did before Volta, so unlike
//! the AMDGPU side's scheduling-only barrier this has to be a real reconvergence:
//! `bar.warp.sync` is what `__syncwarp()` compiles to, and it likewise orders memory among the
//! lanes that take part.
//!
//! The compiler is held back by the intrinsics themselves rather than by a fence: neither is
//! declared `IntrNoMem` in `IntrinsicsNVVM.td`, so both read and write unmodelled memory and
//! nothing moves across them.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
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
    barrier(scope, BARRIER_WARP, FULL_MASK);
}

/// Blocks until every unit of the cube reached this point, and makes what each of them wrote
/// before it visible to all the others.
pub fn lower_sync_cube(scope: &Scope) {
    barrier(scope, BARRIER_CTA, BARRIER_ID);
}
