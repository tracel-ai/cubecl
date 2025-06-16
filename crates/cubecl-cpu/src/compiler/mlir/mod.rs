pub mod builtin;
pub(super) mod external_function;
pub(super) mod memref;
pub mod module;
pub mod transformer;
pub(super) mod visitor;

use builtin::Builtin;
use cubecl_opt::Optimizer;
use external_function::register_external_function;
use memref::LineMemRef;
pub use visitor::elem::register_supported_types;

use std::fmt::{Debug, Display};

use cubecl_core::{prelude::KernelDefinition, server::ScalarBinding};
use module::Module;
use tracel_llvm::melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};

const MAX_BUFFER_SIZE: usize = 256;

pub struct MlirEngine {
    args_zero_indirection: Vec<LineMemRef>,
    args_first_indirection: Vec<*mut ()>,
    args_second_indirection: Vec<*mut ()>,
    pub builtin: Builtin,
    scalars: Vec<ScalarBinding>,
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
    pub fn from_cubecl_ir(kernel: KernelDefinition, opt: &Optimizer) -> Self {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        register_all_passes();

        let context = Context::new();
        register_all_llvm_translations(&context);
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context);

        module.visit_kernel(&kernel, opt);

        module.run_pass();

        let execution_engine = module.into_execution_engine();
        let args_zero_indirection = Vec::with_capacity(MAX_BUFFER_SIZE);
        let args_first_indirection = Vec::with_capacity(MAX_BUFFER_SIZE);
        let args_second_indirection = Vec::with_capacity(MAX_BUFFER_SIZE);
        let scalars = Vec::with_capacity(MAX_BUFFER_SIZE);
        let mut builtin = Builtin::default();
        builtin.set_cube_dim(kernel.cube_dim);
        register_external_function(&execution_engine);
        Self {
            execution_engine,
            args_zero_indirection,
            args_first_indirection,
            args_second_indirection,
            scalars,
            builtin,
        }
    }

    pub fn dump_object(&self, path: &str) {
        self.execution_engine.dump_to_object_file(path);
    }

    /// # Safety
    /// This function will make the program segfault if args is reallocated.
    pub unsafe fn push_buffer(&mut self, pointer: &mut [u8]) {
        let first_box = LineMemRef::new(pointer);
        self.args_zero_indirection.push(first_box);
        let undirected = self.args_zero_indirection.last_mut().unwrap() as *mut LineMemRef;
        self.args_first_indirection.push(undirected as *mut ());
        let undirected = self.args_first_indirection.last_mut().unwrap() as *mut *mut ();
        self.args_second_indirection.push(undirected as *mut ());
    }

    pub fn push_scalar(&mut self, scalar: ScalarBinding) {
        self.scalars.push(scalar);
        let data = self.scalars.last_mut().unwrap().data.as_mut_ptr() as *mut u8;
        self.args_second_indirection.push(data as *mut ());
    }

    pub fn push_builtin(&mut self) {
        for arg in self.builtin.dims.iter_mut() {
            self.args_second_indirection
                .push(arg as *mut u64 as *mut ());
        }
    }

    /// # Safety
    /// MLIR kernel needs valid reference and will SEGFAULT if bad pointer are sent.
    pub unsafe fn run_kernel(&mut self) {
        unsafe {
            self.execution_engine
                .invoke_packed("kernel", &mut self.args_second_indirection)
                .unwrap()
        }
    }
}
