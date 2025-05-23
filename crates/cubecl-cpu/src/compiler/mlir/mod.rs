pub(super) mod elem;
pub(super) mod item;
pub mod module;
pub(super) mod operation;
pub(super) mod operator;
pub(super) mod scope;
pub(super) mod visitor;

pub use elem::register_supported_types;

use std::fmt::{Debug, Display};

use cubecl_core::prelude::KernelDefinition;
use melior::{Context, ExecutionEngine, dialect::DialectRegistry, utility::register_all_dialects};
use module::Module;

const MAX_BUFFER_SIZE: usize = 16;

pub struct MlirEngine {
    // This field must never be reallocated, because a double indirection is necessary for Orca JIT
    args: Vec<*mut u8>,
    args_indirected: Vec<*mut ()>,
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
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context);

        module.visit_kernel(&kernel);

        module.run_pass();

        let execution_engine = module.into_execution_engine();
        let args = Vec::with_capacity(MAX_BUFFER_SIZE);
        let args_indirected = Vec::with_capacity(MAX_BUFFER_SIZE);
        Self {
            execution_engine,
            args,
            args_indirected,
        }
    }

    pub fn dump_object(&self, path: &str) {
        self.execution_engine.dump_to_object_file(path);
    }

    /// This function will make the program segfault if args is reallocated
    pub unsafe fn push_buffer(&mut self, ptr: &mut [u8]) {
        self.args.push(ptr.as_mut_ptr());
        let last_elem = unsafe { self.args.last_mut().unwrap_unchecked() };
        self.args_indirected
            .push(last_elem as *mut *mut u8 as *mut ());
    }

    pub unsafe fn run_kernel(&mut self) {
        unsafe {
            self.execution_engine
                .invoke_packed("kernel", &mut self.args_indirected)
                .unwrap()
        }
    }
}
