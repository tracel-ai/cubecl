pub mod builtin;
pub(super) mod external_function;
pub(super) mod memref;
pub mod mlir_data;
pub mod mlir_engine;
pub mod module;
pub mod passes;
pub(super) mod visitor;

use passes::shared_memories::SharedMemories;
pub use visitor::elem::register_supported_types;

use cubecl_core::{
    Compiler, ExecutionMode,
    ir::{self},
    prelude::KernelDefinition,
};
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
        #[cfg(feature = "mlir-dump")]
        dump_scope(&kernel.body);
        let opt = OptimizerBuilder::default()
            .with_transformer(ErfTransform)
            .optimize(kernel.body.clone(), kernel.cube_dim, mode);

        let mut shared_memories = SharedMemories::default();
        shared_memories.visit(&opt);

        #[cfg(feature = "mlir-dump")]
        dump_opt(&opt);
        MlirEngine::from_cubecl_ir(kernel, &opt, shared_memories)
    }

    fn elem_size(&self, elem: ir::Elem) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_scope(scope: &cubecl_core::prelude::Scope) {
    use std::fs;

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_MLIR") {
        fs::write(format!("{dir}/cubecl.ir.txt"), format!("{}", scope)).unwrap();
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_opt(opt: &cubecl_opt::Optimizer) {
    use std::fs;

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_MLIR") {
        fs::write(format!("{dir}/cubecl-opt.ir.txt"), format!("{}", opt)).unwrap();
        fs::write(
            format!("{dir}/cubecl-opt.ir.dot"),
            format!("{}", opt.dot_viz()),
        )
        .unwrap();
    }
}
