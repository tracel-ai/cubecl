pub mod module;

use std::fmt::{Debug, Display};

use cubecl_core::prelude::KernelDefinition;
use melior::{Context, ExecutionEngine, dialect::DialectRegistry, utility::register_all_dialects};
use module::Module;

pub struct MlirEngine {
    execution_engine: ExecutionEngine,
}

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
    pub fn from_cubecl_ir(kernel: KernelDefinition) -> Self {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context);

        module.visit_kernel(&kernel);

        module.run_pass();

        let execution_engine = module.into_execution_engine();
        Self { execution_engine }
    }

    pub fn run_kernel(&self) {
        unsafe {
            self.execution_engine
                .invoke_packed("kernel", &mut [])
                .unwrap()
        }
    }
}
