pub mod kernel;

use cubecl_core::Compiler;
use kernel::MLIRKernel;

#[derive(Debug, Clone)]
pub struct MLIRCompiler {}

#[derive(Default, Debug)]
pub struct MLIRCompilerOptions {}

impl Compiler for MLIRCompiler {
    type Representation = MLIRKernel;

    type CompilationOptions = MLIRCompilerOptions;

    fn compile(
        &mut self,
        kernel: cubecl_core::prelude::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: cubecl_core::ExecutionMode,
    ) -> Self::Representation {
        todo!()
    }

    fn elem_size(&self, elem: cubecl_core::ir::Elem) -> usize {
        todo!()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
