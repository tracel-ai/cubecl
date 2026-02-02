use cubecl_core::{ir::StorageType, prelude::KernelDefinition};
use cubecl_opt::Optimizer;
use tracel_llvm::mlir_rs::{
    Context, ExecutionEngine,
    ir::{Location, operation::OperationLike},
    pass::{self, PassManager},
};

use super::{passes::shared_memories::SharedMemories, visitor::Visitor};

pub(super) struct Module<'a> {
    module: tracel_llvm::mlir_rs::ir::Module<'a>,
    #[allow(unused)]
    name: String,
    location: Location<'a>,
    context: &'a Context,
}

impl<'a> Module<'a> {
    pub(super) fn new(context: &'a Context, name: String) -> Self {
        let location = Location::unknown(context);
        let module = tracel_llvm::mlir_rs::ir::Module::new(location);
        Self {
            module,
            context,
            name,
            location,
        }
    }

    pub(super) fn visit_kernel(
        &mut self,
        kernel: &KernelDefinition,
        opt: &Optimizer,
        shared_memories: &SharedMemories,
        addr_type: StorageType,
    ) {
        Visitor::visit_kernel(
            self.context,
            self.location,
            kernel,
            &self.module,
            opt,
            shared_memories,
            addr_type,
        )
    }

    pub(super) fn run_pass(&mut self) {
        let pass_manager = PassManager::new(self.context);
        pass_manager.enable_verifier(true);
        #[cfg(feature = "mlir-dump")]
        if let Ok(dir) = std::env::var("CUBECL_DEBUG_MLIR") {
            use std::path::PathBuf;
            use tracel_llvm::mlir_rs::{
                ir::operation::OperationPrintingFlags, pass::PassIrPrintingOptions,
            };

            let dir = dir.to_string() + "/" + &self.name;
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
        pass_manager.add_pass(pass::conversion::create_math_to_libm());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        // Clean up unrealized casts now so cf.cond_br block arguments don't carry memref types
        // into cf-to-llvm (which would fail verification).
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
        // scf-to-cf can introduce index-typed block arguments (e.g. loop induction variables).
        // Run index/math/vector/arith lowering inside func scope before cf-to-llvm.
        let func_passes = pass_manager.nested_under("func.func");
        func_passes.add_pass(pass::conversion::create_index_to_llvm());
        func_passes.add_pass(pass::conversion::create_math_to_llvm());
        func_passes.add_pass(pass::conversion::create_vector_to_llvm());
        func_passes.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
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

#[cfg(test)]
mod tests {
    use tracel_llvm::mlir_rs::{
        Context,
        dialect::{arith, func, index, memref, scf},
        ir::{
            Block, Region, RegionLike, Type,
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::{FunctionType, IntegerType, MemRefType},
        },
        utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    };

    /// Regression test for MLIR lowering failure:
    ///   `llvm.cond_br` operand must be LLVM dialect-compatible, but got `memref<...>`
    ///
    /// Reproduces the issue: `scf.if` yields a memref result, which after scf-to-cf becomes
    /// a `cf.cond_br` with memref-typed block arguments. If cf-to-llvm runs before memrefs
    /// are lowered, verification fails. The fix is to run finalize-memref-to-llvm and
    /// reconcile-unrealized-casts between scf-to-cf and cf-to-llvm.
    #[test]
    fn it_lowers_scf_if_memref_result_through_cf_to_llvm() {
        register_all_passes();
        let registry = tracel_llvm::mlir_rs::dialect::DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        register_all_llvm_translations(&context);
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = super::Module::new(&context, "scf_if_memref_result".to_string());
        let location = module.location;

        let memref_ty = MemRefType::new(Type::bfloat16(&context), &[8], None, None);
        let i32_ty: Type = IntegerType::new(&context, 32).into();

        module.module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "kernel"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                let c0 = block.append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(i32_ty, 0).into(),
                    location,
                ));
                let c1 = block.append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(i32_ty, 1).into(),
                    location,
                ));
                let cond = block.append_operation(arith::cmpi(
                    &context,
                    arith::CmpiPredicate::Ne,
                    c0.result(0).unwrap().into(),
                    c1.result(0).unwrap().into(),
                    location,
                ));

                let a = block.append_operation(memref::alloca(
                    &context,
                    memref_ty,
                    &[],
                    &[],
                    None,
                    location,
                ));
                let b = block.append_operation(memref::alloca(
                    &context,
                    memref_ty,
                    &[],
                    &[],
                    None,
                    location,
                ));

                let then_region = {
                    let then_block = Block::new(&[]);
                    then_block
                        .append_operation(scf::r#yield(&[a.result(0).unwrap().into()], location));
                    let region = Region::new();
                    region.append_block(then_block);
                    region
                };

                let else_region = {
                    let else_block = Block::new(&[]);
                    else_block
                        .append_operation(scf::r#yield(&[b.result(0).unwrap().into()], location));
                    let region = Region::new();
                    region.append_block(else_block);
                    region
                };

                let if_op = block.append_operation(scf::r#if(
                    cond.result(0).unwrap().into(),
                    &[memref_ty.into()],
                    then_region,
                    else_region,
                    location,
                ));

                // Use the yielded memref so the `scf.if` can't be trivially DCE'd.
                let idx0 = block.append_operation(index::constant(
                    &context,
                    IntegerAttribute::new(IntegerType::new(&context, 64).into(), 0),
                    location,
                ));
                let _ = block.append_operation(memref::load(
                    if_op.result(0).unwrap().into(),
                    &[idx0.result(0).unwrap().into()],
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        module.run_pass();
        assert!(module.module.as_operation().verify());
    }
}
