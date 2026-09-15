//! Atomics with explicit memory ordering.

use cubecl_core::ir::dialect::atomic::AtomicLoadOp;
use cubecl_core::ir::dialect::memory::LoadOp;
use cubecl_core::ir::dialect::plane::{AtomicUniformLoadOp, UniformLoadOp};
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope, SyncScopeAttr};
use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use pliron_llvm::attributes::AtomicOrderingAttr;

use crate::shared::polyfill::LowerOp;

#[cube_op(name = "cpu.ordered_atomic_load")]
#[result_ty(argument)]
pub struct OrderedAtomicLoadOp {
    pub ptr: Value,
    pub ordering: AtomicOrderingAttr,
}

#[cube_op(name = "cpu.ordered_atomic_store")]
#[result_ty(none)]
pub struct OrderedAtomicStoreOp {
    pub ptr: Value,
    pub value: Value,
    pub ordering: AtomicOrderingAttr,
}

#[cube_op(name = "cpu.ordered_atomic_fetch_add")]
#[result_ty(same_as = value)]
pub struct OrderedAtomicFetchAddOp {
    pub ptr: Value,
    pub value: Value,
    pub ordering: AtomicOrderingAttr,
}

/// Acquire load.
#[cube]
pub fn atomic_load_acquire(atomic: &Atomic<u32>) -> u32 {
    intrinsic!(|scope| {
        let ptr = atomic.value(scope);
        let ty = u32::__expand_as_type(scope);
        let op = OrderedAtomicLoadOp::new(scope.ctx_mut(), ty, ptr, AtomicOrderingAttr::Acquire);
        scope.register_with_result(&op).into()
    })
}

/// Release store.
#[cube]
pub fn atomic_store_release(atomic: &Atomic<u32>, value: u32) {
    intrinsic!(|scope| {
        let ptr = atomic.value(scope);
        let value = value.read_value(scope);
        let op =
            OrderedAtomicStoreOp::new(scope.ctx_mut(), ptr, value, AtomicOrderingAttr::Release);
        scope.register(&op);
    })
}

/// Acquire-release addition. Returns the previous value.
#[cube]
pub fn atomic_fetch_add_acq_rel(atomic: &Atomic<u32>, value: u32) -> u32 {
    intrinsic!(|scope| {
        let ptr = atomic.value(scope);
        let value = value.read_value(scope);
        let op =
            OrderedAtomicFetchAddOp::new(scope.ctx_mut(), ptr, value, AtomicOrderingAttr::AcqRel);
        scope.register_with_result(&op).into()
    })
}

#[op_interface_impl]
impl LowerOp for UniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))]
    }
}

#[op_interface_impl]
impl LowerOp for AtomicUniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&AtomicLoadOp::new(scope.ctx_mut(), ptr))]
    }
}
