pub mod kernel;
pub mod mlir_converter;
pub mod scope;
pub mod supported_types;

use cubecl_core::{Compiler, ExecutionMode, ir, prelude::KernelDefinition};
use kernel::MLIRKernel;

#[derive(Debug, Clone, Default)]
pub struct MLIRCompiler {}

#[derive(Default, Debug)]
pub struct MLIRCompilerOptions {}

impl Compiler for MLIRCompiler {
    type Representation = MLIRKernel;

    type CompilationOptions = MLIRCompilerOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        println!("{}", kernel.body);
        self.visit(&kernel.body);
        MLIRKernel
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
