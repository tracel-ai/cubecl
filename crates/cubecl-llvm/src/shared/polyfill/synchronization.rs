//! Dispatch of `sync.sync` to the target that implements it.
//!
//! An op carries one implementation of an interface, so the two barriers live with their
//! targets — [`amdgpu`](crate::amdgpu::synchronization), [`cpu`](crate::cpu::synchronization) —
//! and this chooses between them.

use cubecl_core::ir::Scope;
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope};
use cubecl_core::ir::prelude::*;

use crate::shared::polyfill::LowerOp;
use crate::target::{CtxTarget, LlvmTarget};

#[op_interface_impl]
impl LowerOp for SyncOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        // Read out before lowering: holding the attribute borrowed clashes with the ops built
        // under it.
        let sync_scope = self.scope(ctx).0;
        let target = ctx.target();
        let op = self.get_operation();

        match sync_scope {
            // A plane is one unit on the CPU, so it is always in sync with itself. On the GPU
            // it is a wavefront, which executes in lockstep for the same reason.
            SyncScope::Plane => {}
            SyncScope::Cube => match target {
                LlvmTarget::Cpu => crate::cpu::synchronization::lower_sync_cube(scope, op),
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_cube(scope),
            },
            SyncScope::Device => match target {
                LlvmTarget::Cpu => {
                    panic!("Device wide synchronization is not supported by the CPU runtime")
                }
                LlvmTarget::AmdGpu => {
                    unimplemented!(
                        "device wide synchronization is not supported on the AMDGPU target yet"
                    )
                }
            },
            SyncScope::Unit => {}
        }
        vec![]
    }
}
