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
        // let module = melior::ir::Module::parse(
        //     context,
        //     r#"
        //     module {
        //         func.func @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> f32 attributes {llvm.emit_c_interface} {
        //             %c2 = arith.constant 1 : index
        //             %3 = memref.load %arg0[%c2] : memref<?xf32>
        //             %4 = memref.load %arg1[%c2] : memref<?xf32>
        //             %res = arith.addf %3, %4 : f32
        //             memref.store %res, %arg2[%c2] : memref<?xf32>
        //             return %res : f32
        //         }
        //     }
        //     "#,
        // ).unwrap();
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
            before_all: true,
            after_all: true,
            module_scope: true,
            on_change: true,
            on_failure: true,
            flags: OperationPrintingFlags::new(),
            tree_printing_path: PathBuf::from("debug"),
        });

        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
        pass_manager.add_pass(pass::transform::create_mem_2_reg());
        pass_manager.add_pass(pass::transform::create_inliner());
        pass_manager.add_pass(pass::transform::create_remove_dead_values());
        pass_manager.add_pass(pass::transform::create_generate_runtime_verification());
        pass_manager.run(&mut self.module).unwrap();
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(&self) -> ExecutionEngine {
        let engine = ExecutionEngine::new(&self.module, 3, &[], true);
        engine.dump_to_object_file("test.so");
        engine
    }
}
