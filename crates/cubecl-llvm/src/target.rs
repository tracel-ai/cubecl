//! LLVM compilation targets.

use crate::prelude::{Context, ContextExt};

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
