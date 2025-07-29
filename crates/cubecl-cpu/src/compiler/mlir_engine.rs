use crate::compiler::mlir_data::MlirData;

use super::{
    external_function::register_external_function, passes::shared_memories::SharedMemories,
};
use cubecl_opt::Optimizer;

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use super::module::Module;
use cubecl_core::prelude::KernelDefinition;
use tracel_llvm::melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};

pub struct MlirKernel {
    execution_engine: ExecutionEngine,
    pub shared_memories: SharedMemories,
}

#[derive(Clone)]
pub struct MlirEngine(pub Arc<MlirKernel>);

impl Debug for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Work in progress")
    }
}

impl Display for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "work in progress")
    }
}

impl MlirEngine {
    pub fn from_cubecl_ir(
        kernel: KernelDefinition,
        opt: &Optimizer,
        shared_memories: SharedMemories,
    ) -> Self {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        register_all_passes();

        let context = Context::new();
        register_all_llvm_translations(&context);
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context);

        module.visit_kernel(&kernel, opt, &shared_memories);

        module.run_pass();

        let execution_engine = module.into_execution_engine();
        register_external_function(&execution_engine);
        let kernel = MlirKernel {
            execution_engine,
            shared_memories,
        };
        let mlir_kernel = Arc::new(kernel);
        Self(mlir_kernel)
    }

    pub fn dump_object(&self, path: &str) {
        self.0.execution_engine.dump_to_object_file(path);
    }

    /// # Safety
    /// MLIR kernel needs valid reference to memory and will segfault if bad pointer are sent.
    #[inline(always)]
    pub unsafe fn run_kernel(&mut self, mlir_data: &mut MlirData) {
        unsafe {
            self.0
                .execution_engine
                .invoke_packed("kernel", &mut mlir_data.args_second_indirection)
                .unwrap()
        }
    }
}
