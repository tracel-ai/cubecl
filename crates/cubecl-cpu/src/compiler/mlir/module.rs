/// This module emulate all CubeCL constant using the equivalent MLIR skeleton on top of the kernel
/// ```
/// let module = melior::ir::Module::parse(
///     context,
///     r#"
///     module {
///         func.func private @print_i(index)
///
///         func.func @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %cube_dim_x: index, %cube_dim_y: index, %cube_dim_z: index, %cube_count_x: index, %cube_count_y: index, %cube_count_z: index, %unit_pos_x: index, %unit_pos_y: index, %unit_pos_z: index) attributes {llvm.emit_c_interface} {
///             %cc1 = arith.constant 1 : index
///             %cc0 = arith.constant 0 : index
///
///             %absolute_pos_tmp0 = arith.muli %cube_count_x, %cube_dim_x : index
///             %absolute_pos_tmp1 = arith.muli %cube_count_y, %cube_dim_y : index
///             %absolute_pos_tmp2 = arith.muli %absolute_pos_tmp0, %absolute_pos_tmp1 : index
///
///             scf.for %cube_pos_x = %cc0 to %cube_count_x step %cc1 {
///                 %absolute_pos_x0 = arith.muli %cube_pos_x, %cube_dim_x : index
///                 %absolute_pos_x1 = arith.addi %absolute_pos_x0, %unit_pos_x : index
///
///                 scf.for %cube_pos_y = %cc0 to %cube_count_y step %cc1 {
///                     %absolute_pos_y0 = arith.muli %cube_pos_y, %cube_dim_y : index
///                     %absolute_pos_y1 = arith.addi %absolute_pos_y0, %unit_pos_y : index
///                     %absolute_pos_tmp3 = arith.muli %absolute_pos_y1, %absolute_pos_tmp0 : index
///
///                     scf.for %cube_pos_z = %cc0 to %cube_count_z step %cc1 {
///                         %absolute_pos_z0 = arith.muli %cube_pos_z, %cube_dim_z : index
///                         %absolute_pos_z1 = arith.addi %absolute_pos_z0, %unit_pos_z : index
///
///                         %absolute_pos_tmp4 = arith.muli %absolute_pos_z1, %absolute_pos_tmp2 : index
///
///                         %absolute_pos_tmp5 = arith.addi %absolute_pos_x1, %absolute_pos_tmp3 : index
///                         %absolute_pos = arith.addi %absolute_pos_tmp5, %absolute_pos_tmp4 : index
///
///                         %cc16 = arith.constant 16 : index
///                         %absolute_pos_x16 = arith.muli %cc16, %absolute_pos : index
///
///                         %0 = vector.load %arg0[%absolute_pos_x16] : memref<?xf32>, vector<16xf32>
///                         %1 = vector.load %arg1[%absolute_pos_x16] : memref<?xf32>, vector<16xf32>
///                         %2 = arith.addf %0, %1 : vector<16xf32>
///                         vector.store %2, %arg2[%absolute_pos_x16] : memref<?xf32>, vector<16xf32>
///                     }
///                 }
///             }
///             func.return
///         }
///     }
///     "#,
/// ).unwrap();
/// ```
use std::path::PathBuf;

use cubecl_core::prelude::KernelDefinition;
use cubecl_opt::Optimizer;
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

    pub(super) fn visit_kernel(&mut self, kernel: &KernelDefinition, opt: &Optimizer) {
        Visitor::new(self.context, self.location).visit_kernel(kernel, &self.module, opt);
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
        pass_manager.run(&mut self.module).unwrap();
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(&self) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, 0, &[], true)
    }
}
