pub mod mlir;

use cubecl_core::{Compiler, ExecutionMode, ir, prelude::KernelDefinition};
use cubecl_opt::OptimizerBuilder;
use mlir::MlirEngine;

<<<<<<< HEAD
#[derive(Clone, Debug, Default)]
pub struct MlirCompiler {}
=======
#[derive(Debug, Clone, Default)]
pub struct MLIRCompiler {}
>>>>>>> e46fd3f9 (feat: add display for scope)

#[derive(Default, Debug)]
pub struct MlirCompilerOptions {}

impl Compiler for MlirCompiler {
    type Representation = MlirEngine;

    type CompilationOptions = MlirCompilerOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions, // TODO pass this through the visitor, though it doesn't need anything for the moment
        mode: ExecutionMode, // TODO support this by adding array bound checking
    ) -> Self::Representation {
<<<<<<< HEAD
        let opt = OptimizerBuilder::default().optimize(kernel.body.clone(), kernel.cube_dim, mode);
        kernel.body = opt.root_scope;
        MlirEngine::from_cubecl_ir(kernel)
=======
        println!("{}", kernel.body);
        MLIRKernel
>>>>>>> e46fd3f9 (feat: add display for scope)
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}
