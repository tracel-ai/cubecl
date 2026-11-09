//! Target-specific synchronization.

use crate::prelude::*;
use cubecl_core::ir::dialect::synchronization::{
    FenceOp, MemoryOrder, SyncAsyncProxyOp, SyncOp, SyncPlaneStorageOp, SyncScope,
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
            // The same, with the release and acquire reaching the host and peer devices.
            SyncScope::System => match target {
                LlvmTarget::Cpu => {
                    panic!("System wide synchronization is not supported by the CPU runtime")
                }
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_system(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_system(scope),
            },
        }
        vec![]
    }
}

/// The plane barrier: `bar.warp.sync` and the wavefront fence order every address space among
/// the plane's units already, storage included.
#[op_interface_impl]
impl LowerOp for SyncPlaneStorageOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        match scope.ctx().target() {
            LlvmTarget::Cpu => {}
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_plane(scope),
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_plane(scope),
        }
        vec![]
    }
}

/// Only NVPTX has an async proxy to fence; elsewhere shared memory has one view and the fence is
/// nothing.
#[op_interface_impl]
impl LowerOp for SyncAsyncProxyOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        match scope.ctx().target() {
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_async_proxy(scope),
            _ => {}
        }
        vec![]
    }
}

/// LLVM's `fence` at the target's name for the scope; a scope a target has no name for takes the
/// next wider one it has. At unit scope there is nothing to order.
#[op_interface_impl]
impl LowerOp for FenceOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let ordering = match self.order(ctx).0 {
            MemoryOrder::Acquire => AtomicOrderingAttr::Acquire,
            MemoryOrder::Release => AtomicOrderingAttr::Release,
            MemoryOrder::AcqRel => AtomicOrderingAttr::AcqRel,
        };
        let named = |name: &str| SyncScopeAttr::NamedScope(name.into());
        let sync_scope = match (ctx.target(), self.scope(ctx).0) {
            (_, SyncScope::Unit) => return vec![],
            (_, SyncScope::System) | (LlvmTarget::Cpu, _) => SyncScopeAttr::System,
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Plane) => named("wavefront"),
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Cube) => named("workgroup"),
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Device) => named("agent"),
            #[cfg(feature = "nvptx")]
            (LlvmTarget::Nvptx, SyncScope::Plane | SyncScope::Cube) => named("block"),
            #[cfg(feature = "nvptx")]
            (LlvmTarget::Nvptx, SyncScope::Device) => named("device"),
        };
        let fence = llvm::FenceOp::new(scope.ctx_mut(), ordering, sync_scope);
        scope.register(&fence);
        vec![]
    }
}
