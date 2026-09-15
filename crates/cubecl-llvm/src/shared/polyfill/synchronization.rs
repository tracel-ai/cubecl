//! Target-specific synchronization.

use cubecl_core::ir::Scope;
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope};
use cubecl_core::ir::prelude::*;

use crate::shared::polyfill::LowerOp;
use crate::target::{CtxTarget, LlvmTarget};

#[op_interface_impl]
impl LowerOp for SyncOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let sync_scope = self.scope(ctx).0;
        let target = ctx.target();
        let op = self.get_operation();

        match sync_scope {
            SyncScope::Unit => {}
            SyncScope::Plane => match target {
                LlvmTarget::Cpu => {}
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_plane(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_plane(scope),
            },
            // GPU synchronization is limited to the current cube.
            SyncScope::Cube | SyncScope::Device => match target {
                LlvmTarget::Cpu if sync_scope == SyncScope::Device => {
                    panic!("Device wide synchronization is not supported by the CPU runtime")
                }
                LlvmTarget::Cpu => crate::cpu::synchronization::lower_sync_cube(scope, op),
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_cube(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_cube(scope),
            },
        }
        vec![]
    }
}
