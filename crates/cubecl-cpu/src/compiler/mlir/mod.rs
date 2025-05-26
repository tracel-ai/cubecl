pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod memref;
pub mod module;
pub(super) mod operation;
pub(super) mod operator;
pub(super) mod scope;
pub(super) mod variable;
pub(super) mod visitor;

pub use elem::register_supported_types;
use memref::LineMemRef;

use std::fmt::{Debug, Display};

use cubecl_core::prelude::KernelDefinition;
use melior::{Context, ExecutionEngine, dialect::DialectRegistry, utility::register_all_dialects};
use module::Module;

const MAX_BUFFER_SIZE: usize = 16;

pub struct MlirEngine {
    args: Vec<Box<LineMemRef>>,
    args_undirected: Vec<*mut ()>,
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
        let args_undirected = Vec::with_capacity(MAX_BUFFER_SIZE);
        Self {
            execution_engine,
            args,
            args_undirected,
        }
    }

    pub fn dump_object(&self, path: &str) {
        self.execution_engine.dump_to_object_file(path);
    }

    /// This function will make the program segfault if args is reallocated
    pub unsafe fn push_buffer(&mut self, pointer: &mut [u8]) {
        self.args.push(Box::new(LineMemRef::new(pointer)));
        let last_elem = self.args.last_mut().unwrap().as_mut() as *mut LineMemRef;
        self.args_undirected.push(last_elem as *mut ());
    }

    pub unsafe fn run_kernel(&mut self) {
        unsafe {
            self.execution_engine
                .invoke_packed("kernel", &mut self.args_undirected)
                .unwrap()
        }
    }
}
