//! LLVM compilation targets.

use crate::prelude::{Context, ContextExt, SyncScopeAttr};
use cubecl_core::ir::dialect::synchronization::SyncScope;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LlvmTarget {
    #[default]
    Cpu,
    #[cfg(feature = "amdgpu")]
    AmdGpu,
    #[cfg(feature = "nvptx")]
    Nvptx,
}

impl LlvmTarget {
    /// Whether the target provides a hardware launch grid.
    pub fn is_gpu(self) -> bool {
        !matches!(self, LlvmTarget::Cpu)
    }

    /// LLVM's syncscope for `scope` on this target. A scope the target has no name for takes the
    /// next wider one it has.
    pub fn sync_scope(self, scope: SyncScope) -> SyncScopeAttr {
        match (self, scope) {
            (_, SyncScope::Unit) => SyncScopeAttr::SingleThread,
            (_, SyncScope::System) | (LlvmTarget::Cpu, _) => SyncScopeAttr::System,
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Plane) => SyncScopeAttr::NamedScope("wavefront".into()),
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Cube) => SyncScopeAttr::NamedScope("workgroup".into()),
            #[cfg(feature = "amdgpu")]
            (LlvmTarget::AmdGpu, SyncScope::Device) => SyncScopeAttr::NamedScope("agent".into()),
            #[cfg(feature = "nvptx")]
            (LlvmTarget::Nvptx, SyncScope::Plane | SyncScope::Cube) => {
                SyncScopeAttr::NamedScope("block".into())
            }
            #[cfg(feature = "nvptx")]
            (LlvmTarget::Nvptx, SyncScope::Device) => SyncScopeAttr::NamedScope("device".into()),
        }
    }
}

impl CtxTarget for Context {}

/// Minimum CPU buffer alignment, including view offsets.
pub(crate) struct CpuBufferAlignment(pub u32);

pub trait CtxTarget: ContextExt {
    fn target(&self) -> LlvmTarget {
        *self.aux_ty::<LlvmTarget>()
    }
    fn set_target(&mut self, value: LlvmTarget) {
        self.set_aux_ty(value);
    }
}
