pub mod builtin;
pub(super) mod external_function;
pub(super) mod memref;
pub mod mlir_data;
pub mod mlir_engine;
pub mod module;
pub mod passes;
pub(super) mod visitor;

pub use visitor::elem::register_supported_types;

use cubecl_core::{Compiler, ExecutionMode, ir, prelude::KernelDefinition};
use cubecl_opt::OptimizerBuilder;
use mlir_engine::MlirEngine;

use crate::compiler::passes::erf_transform::ErfTransform;

#[derive(Clone, Debug, Default)]
pub struct MlirCompiler {}

#[derive(Default, Debug)]
pub struct MlirCompilerOptions {}

impl Compiler for MlirCompiler {
    type Representation = MlirEngine;

    type CompilationOptions = MlirCompilerOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions, // TODO pass this through the visitor, though it doesn't need anything for the moment
        mode: ExecutionMode, // TODO support this by adding array bound checking
    ) -> Self::Representation {
        let opt = OptimizerBuilder::default()
            .with_transformer(ErfTransform)
            .optimize(kernel.body.clone(), kernel.cube_dim, mode);
        MlirEngine::from_cubecl_ir(kernel, &opt)
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
