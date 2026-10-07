use std::fmt::Debug;
use std::sync::Arc;

use cubecl_server::kernel::CompiledKernel;

use cubecl_llvm::PlironCompiler;

/// A compiled cpu kernel.
#[derive(Clone)]
pub struct CpuCompiledKernel {
    pub(crate) mlir: Arc<CompiledKernel<PlironCompiler>>,
}

impl CpuCompiledKernel {
    pub fn new(kernel: Arc<CompiledKernel<PlironCompiler>>) -> Self {
        Self { mlir: kernel }
    }
}

impl Debug for CpuCompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuCompiledKernel")
            .field("entrypoint_name", &self.mlir.entrypoint_name)
            .field("debug_name", &self.mlir.debug_name)
            .finish()
    }
}
