use std::fmt::Debug;
use std::sync::Arc;

use cubecl_core::prelude::CompiledKernel;

use crate::compiler::PlironCompiler;

/// A compiled cpu kernel.
pub struct CpuKernel {
    pub(crate) mlir: Arc<CompiledKernel<PlironCompiler>>,
}

impl CpuKernel {
    pub fn new(kernel: CompiledKernel<PlironCompiler>) -> Self {
        Self {
            mlir: Arc::new(kernel),
        }
    }
}

impl Debug for CpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuKernel")
            .field("entrypoint_name", &self.mlir.entrypoint_name)
            .field("debug_name", &self.mlir.debug_name)
            .finish()
    }
}
