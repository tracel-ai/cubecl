use cubecl_core::prelude::KernelDefinition;
use melior::{
    Context, ExecutionEngine,
    dialect::func,
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Region, RegionLike,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::FunctionType,
    },
    pass::{self, PassManager},
};

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

    pub(super) fn visit_kernel(&mut self, _kernel: &KernelDefinition) {
        let name = StringAttribute::new(self.context, "kernel");
        let mlir_type = TypeAttribute::new(FunctionType::new(self.context, &[], &[]).into());

        let block = Block::new(&[]);

        block.append_operation(func::r#return(&[], self.location));

        let region = Region::new();
        region.append_block(block);

        let attributes = &[(
            Identifier::new(self.context, "llvm.emit_c_interface"),
            Attribute::unit(self.context).into(),
        )];
        self.module.body().append_operation(func::func(
            self.context,
            name,
            mlir_type,
            region,
            attributes,
            self.location,
        ));
    }

    pub(super) fn run_pass(&mut self) {
        let pass_manager = PassManager::new(&self.context);
        pass_manager.enable_verifier(true);
        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.run(&mut self.module).unwrap();
        self.module.as_operation().verify();
    }

    pub(super) fn into_execution_engine(&self) -> ExecutionEngine {
        ExecutionEngine::new(&self.module, 3, &[], true)
    }
}
