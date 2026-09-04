//! Which machine the pliron pipeline is lowering for.
//!
//! Chosen once from the environment and put on the [`Context`], so the passes both pipelines
//! share can ask. The gfx architecture is not part of it: that belongs to the device and
//! arrives later, on [`PlironOptions`](crate::PlironOptions).

use cubecl_core::ir::ContextExt;
use pliron::context::Context;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LlvmTarget {
    #[default]
    Cpu,
    #[cfg(feature = "amdgpu")]
    AmdGpu,
}

impl CtxTarget for Context {}

/// The target on the context, as the passes shared by both pipelines see it.
pub trait CtxTarget: ContextExt {
    fn target(&self) -> LlvmTarget {
        *self.aux_ty::<LlvmTarget>()
    }
    fn set_target(&mut self, value: LlvmTarget) {
        self.set_aux_ty(value);
    }
}
