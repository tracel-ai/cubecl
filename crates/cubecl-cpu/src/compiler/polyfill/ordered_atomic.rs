//! Atomics carrying an explicit memory ordering.
//!
//! The `atomic` cube dialect ops are lowered with monotonic ordering, which is all a counter
//! needs but not what a barrier needs: [`sync_cube`](super::synchronization) has to publish the
//! writes made before it and to acquire the ones made by the other units of the cube. These
//! CPU-private ops carry the ordering they must be lowered with, and become their `llvm` dialect
//! counterpart in [`to_llvm::atomic`](super::to_llvm::atomic).

use cubecl_core::ir::dialect::atomic::AtomicLoadOp;
use cubecl_core::ir::dialect::memory::LoadOp;
use cubecl_core::ir::dialect::plane::{AtomicUniformLoadOp, UniformLoadOp};
use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use pliron_llvm::attributes::AtomicOrderingAttr;

use crate::compiler::polyfill::LowerOp;

/// Atomic load, i.e. `atomic.load` with an explicit ordering.
#[cube_op(name = "cpu.ordered_atomic_load")]
#[result_ty(argument)]
pub struct OrderedAtomicLoadOp {
    pub ptr: Value,
    pub ordering: AtomicOrderingAttr,
}

/// Atomic store, i.e. `atomic.store` with an explicit ordering.
#[cube_op(name = "cpu.ordered_atomic_store")]
#[result_ty(none)]
pub struct OrderedAtomicStoreOp {
    pub ptr: Value,
    pub value: Value,
    pub ordering: AtomicOrderingAttr,
}

/// Atomic add returning the previous value, i.e. `atomic.i_add` with an explicit ordering.
#[cube_op(name = "cpu.ordered_atomic_fetch_add")]
#[result_ty(same_as = value)]
pub struct OrderedAtomicFetchAddOp {
    pub ptr: Value,
    pub value: Value,
    pub ordering: AtomicOrderingAttr,
}

/// Atomically loads `atomic`, acquiring everything the unit that released it wrote before.
#[cube]
pub fn atomic_load_acquire(atomic: &Atomic<u32>) -> u32 {
    intrinsic!(|scope| {
        let ptr = atomic.value(scope);
        let ty = u32::__expand_as_type(scope);
        let op = OrderedAtomicLoadOp::new(scope.ctx_mut(), ty, ptr, AtomicOrderingAttr::Acquire);
        scope.register_with_result(&op).into()
    })
}

/// Atomically stores `value` into `atomic`, releasing everything written before it.
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

/// Atomically adds `value` to `atomic` and returns the previous value, releasing everything
/// written before it and acquiring what the other units released.
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
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))]
    }
}

#[op_interface_impl]
impl LowerOp for AtomicUniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&AtomicLoadOp::new(scope.ctx_mut(), ptr))]
    }
}
