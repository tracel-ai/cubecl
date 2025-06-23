pub mod mlir;

use cubecl_core::{Compiler, ExecutionMode, ir, prelude::KernelDefinition};
use cubecl_opt::OptimizerBuilder;
use mlir::MlirEngine;

use crate::compiler::mlir::passes::{
    checked_transform::CheckedTransform, erf_transform::ErfTransform,
};

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
        let mut opt = OptimizerBuilder::default().with_transformer(ErfTransform);

        if mode == ExecutionMode::Checked {
            opt = opt.with_transformer(CheckedTransform);
        }

        let opt = opt.optimize(kernel.body.clone(), kernel.cube_dim, mode);
        MlirEngine::from_cubecl_ir(kernel, &opt)
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
