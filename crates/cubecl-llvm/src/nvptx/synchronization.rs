//! NVPTX synchronization. CTA and warp barriers also order memory.

use cubecl_core::ir::Scope;
use cubecl_core::ir::prelude::*;
use pliron_llvm::types::VoidType;

use crate::shared::intrinsic::{call_op, i32_const_op};

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
