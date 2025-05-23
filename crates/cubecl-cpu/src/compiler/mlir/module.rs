use std::path::PathBuf;

use cubecl_core::prelude::KernelDefinition;
use melior::{
    Context, ExecutionEngine,
    dialect::{func, llvm},
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Region, RegionLike,
        attribute::{StringAttribute, TypeAttribute},
        operation::{OperationLike, OperationPrintingFlags},
        r#type::FunctionType,
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
        let name = StringAttribute::new(self.context, "kernel");

        let attributes = &[(
            Identifier::new(self.context, "llvm.emit_c_interface"),
            Attribute::unit(self.context).into(),
        )];

        let mut inputs = Vec::with_capacity(kernel.buffers.len());
        let mut block_input = Vec::with_capacity(kernel.buffers.len());

        for _ in kernel.buffers.iter() {
            inputs.push(llvm::r#type::pointer(self.context, 0));
            block_input.push((llvm::r#type::pointer(self.context, 0), self.location));
        }

        let func_type = TypeAttribute::new(FunctionType::new(self.context, &inputs, &[]).into());

        self.module.body().append_operation(func::func(
            self.context,
            name,
            func_type,
            {
                let region = Region::new();
                let block = Block::new(&block_input);

                let visitor = Visitor::new(&block, self.context, self.location);
                visitor.visit_scope(&kernel.body);

                block.append_operation(func::r#return(&[], self.location));

                region.append_block(block);
                region
            },
            attributes,
            self.location,
        ));
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
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.run(&mut self.module).unwrap();
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(&self) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, 3, &[], false)
    }
}
