use cubecl_core::prelude::KernelDefinition;
use cubecl_opt::Optimizer;
use tracel_llvm::melior::{
    Context, ExecutionEngine,
    ir::{Location, operation::OperationLike},
    pass::{self, PassManager},
};

use super::{passes::shared_memories::SharedMemories, visitor::Visitor};

pub(super) struct Module<'a> {
    module: tracel_llvm::melior::ir::Module<'a>,
    location: Location<'a>,
    context: &'a Context,
}

impl<'a> Module<'a> {
    pub(super) fn new(context: &'a Context) -> Self {
        let location = Location::unknown(context);
        let module = tracel_llvm::melior::ir::Module::new(location);
        Self {
            module,
            context,
            location,
        }
    }

    pub(super) fn visit_kernel(
        &mut self,
        kernel: &KernelDefinition,
        opt: &Optimizer,
        shared_memories: &SharedMemories,
    ) {
        Visitor::visit_kernel(
            self.context,
            self.location,
            kernel,
            &self.module,
            opt,
            shared_memories,
        )
    }

    pub(super) fn run_pass(&mut self) {
        let pass_manager = PassManager::new(self.context);
        pass_manager.enable_verifier(true);
        #[cfg(feature = "mlir-dump")]
        if let Ok(dir) = std::env::var("CUBECL_DEBUG_MLIR") {
            use std::path::PathBuf;
            use tracel_llvm::melior::{
                ir::operation::OperationPrintingFlags, pass::PassIrPrintingOptions,
            };

            pass_manager.enable_ir_printing(&PassIrPrintingOptions {
                before_all: true,
                after_all: true,
                module_scope: true,
                on_change: true,
                on_failure: true,
                flags: OperationPrintingFlags::new(),
                tree_printing_path: PathBuf::from(dir),
            });
        }
        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::transform::create_inliner());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
        pass_manager.add_pass(pass::transform::create_sccp());
        pass_manager.add_pass(pass::transform::create_mem_2_reg());
        // pass_manager.add_pass(pass::transform::create_remove_dead_values()); // Needs this to be fixed before https://github.com/llvm/llvm-project/issues/82788
        pass_manager.add_pass(pass::transform::create_control_flow_sink());
        pass_manager.add_pass(pass::transform::create_cse());
        if let Err(err) = pass_manager.run(&mut self.module) {
            panic!("{}", err);
        }
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(self) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, 0, &[], true)
    }
}
