//! Target-specific synchronization.

use crate::prelude::*;
use cubecl_core::ir::dialect::synchronization::{
    AllowDependentKernelsToLaunchOp, SyncOp, SyncScope, WaitForPrerequisiteKernelsOp,
};

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
            SyncScope::Cube => match target {
                LlvmTarget::Cpu => crate::cpu::synchronization::lower_sync_cube(scope, op),
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_cube(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_cube(scope),
            },
            // The units of the cube meet, as at a cube barrier, and what each wrote to storage is
            // visible to every cube of the device, as what other cubes published is visible to it.
            SyncScope::Device => match target {
                LlvmTarget::Cpu => {
                    panic!("Device wide synchronization is not supported by the CPU runtime")
                }
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_storage(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_storage(scope),
            },
        }
        vec![]
    }
}

// Only CUDA launches a kernel before the one ahead of it on the stream has finished, so on the
// other targets there is never anything to wait for or to let start.
#[op_interface_impl]
impl LowerOp for WaitForPrerequisiteKernelsOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        match scope.ctx().target() {
            LlvmTarget::Cpu => {}
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => {}
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => {
                crate::nvptx::synchronization::lower_wait_for_prerequisite_kernels(scope)
            }
        }
        vec![]
    }
}

#[op_interface_impl]
impl LowerOp for AllowDependentKernelsToLaunchOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        match scope.ctx().target() {
            LlvmTarget::Cpu => {}
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => {}
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => {
                crate::nvptx::synchronization::lower_allow_dependent_kernels_to_launch(scope)
            }
        }
        vec![]
    }
}
