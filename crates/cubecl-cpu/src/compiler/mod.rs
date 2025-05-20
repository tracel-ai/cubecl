pub mod mlir;
pub mod supported_types;

use cubecl_core::{Compiler, ExecutionMode, ir, prelude::KernelDefinition};
use mlir::MlirEngine;

#[derive(Clone, Debug, Default)]
pub struct MLIRCompiler {}

#[derive(Default, Debug)]
pub struct MLIRCompilerOptions {}

impl Compiler for MLIRCompiler {
    type Representation = MlirEngine;

    type CompilationOptions = MLIRCompilerOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions, // TODO pass this through the visitor, though it doesn't need anything for the moment
        _mode: ExecutionMode, // TODO support this by adding array bound checking
    ) -> Self::Representation {
        MlirEngine::from_cubecl_ir(kernel)
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
