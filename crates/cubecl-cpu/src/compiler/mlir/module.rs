use std::path::PathBuf;

use cubecl_core::prelude::KernelDefinition;
use melior::{
    Context, ExecutionEngine,
    ir::{
        Location,
        operation::{OperationLike, OperationPrintingFlags},
    },
    pass::{self, PassIrPrintingOptions, PassManager},
};

use super::visitor::Visitor;

pub(super) struct Module<'a> {
    module: melior::ir::Module<'a>,
    location: Location<'a>,
    context: &'a Context,
}

impl<'a> Module<'a> {
    pub(super) fn new(context: &'a Context) -> Self {
        let location = Location::unknown(context);
        let module = melior::ir::Module::new(location);
        Self {
            module,
            context,
            location,
        }
    }

    pub(super) fn visit_kernel(&mut self, kernel: &KernelDefinition) {
        Visitor::new(self.context, self.location).visit_kernel(kernel, &self.module);
    }

    pub(super) fn run_pass(&mut self) {
        let pass_manager = PassManager::new(&self.context);
        pass_manager.enable_verifier(true);
        pass_manager.enable_ir_printing(&PassIrPrintingOptions {
            before_all: false,
            after_all: true,
            module_scope: true,
            on_change: false,
            on_failure: false,
            flags: OperationPrintingFlags::new(),
            tree_printing_path: PathBuf::from("debug"),
        });

        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm()); // This a feature not a bug it
        pass_manager.run(&mut self.module).unwrap();
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(&self) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, 3, &[], false)
    }
}
