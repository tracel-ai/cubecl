use crate::compiler::mlir_data::MlirData;

use super::{
    external_function::register_external_function, passes::shared_memories::SharedMemories,
};

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use super::module::Module;
use cubecl_core::{ir::StorageType, prelude::KernelDefinition};
use cubecl_opt::{Function, GlobalState};
use tracel_llvm::mlir_rs::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};

pub struct MlirKernel {
    execution_engine: ExecutionEngine,
    pub needs_parallelism: bool,
    pub shared_memories: SharedMemories,
}

#[derive(Clone)]
pub struct MlirEngine(pub Arc<MlirKernel>);

impl Debug for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLIR output many IR, so check the README.md on how to generate debug output"
        )
    }
}

impl Display for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLIR output many IR, so check the README.md on how to generate debug output"
        )
    }
}

impl MlirEngine {
    pub fn from_cubecl_ir(
        kernel: KernelDefinition,
        func: &mut Function,
        global_state: &GlobalState,
        shared_memories: SharedMemories,
        addr_type: StorageType,
    ) -> Self {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        register_all_passes();

        let context = Context::new();
        register_all_llvm_translations(&context);
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context, kernel.options.kernel_name.clone());

        module.visit_kernel(&kernel, func, global_state, &shared_memories, addr_type);

        let needs_parallelism = module.needs_parallelism;
        module.run_pass();

        let execution_engine = module.into_execution_engine();
        register_external_function(&execution_engine);
        let mlir_kernel = MlirKernel {
            execution_engine,
            shared_memories,
            needs_parallelism,
        };

        let mlir_kernel = Arc::new(mlir_kernel);
        let mlir_kernel = Self(mlir_kernel);
        #[cfg(feature = "mlir-dump")]
        mlir_kernel.dump_debug_shared_library(&kernel.options.kernel_name);
        mlir_kernel
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
                .unwrap_or_else(|err| {
                    log::error!("MLIR kernel invocation failed: {err}");
                    panic!("{err}");
                });
        }
    }

    #[cfg(feature = "mlir-dump")]
    fn dump_debug_shared_library(&self, name: &str) {
        if let Some(path) = super::get_dump_name(name) {
            self.dump_object(path.join("mlir_output.so").to_str().unwrap());
        }
    }
}
