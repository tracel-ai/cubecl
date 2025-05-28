use std::collections::HashMap;

use cubecl_core::prelude::KernelDefinition;
use cubecl_opt::Optimizer;
use melior::{
    Context,
    dialect::func,
    ir::{
        Attribute, Block, BlockLike, BlockRef, Identifier, Location, Region, RegionLike, Value,
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
    },
};

pub struct Visitor<'a> {
    pub block_stack: Vec<BlockRef<'a, 'a>>,
    pub context: &'a Context,
    pub location: Location<'a>,
    pub current_local_variables: HashMap<u32, Value<'a, 'a>>,
    pub global_buffers: Vec<Value<'a, 'a>>,
}

impl<'a> Visitor<'a> {
    pub fn new(context: &'a Context, location: Location<'a>) -> Self {
        let current_local_variables = HashMap::new();
        let global_buffers = vec![];
        let block_stack = vec![];
        Self {
            block_stack,
            context,
            location,
            current_local_variables,
            global_buffers,
        }
    }

    pub fn block(&self) -> BlockRef<'a, 'a> {
        self.block_stack.last().unwrap().clone()
    }

    pub(super) fn visit_kernel<'b: 'a>(
        &'a mut self,
        kernel: &'b KernelDefinition,
        module: &melior::ir::Module<'a>,
        opt: &Optimizer,
    ) {
        let name = StringAttribute::new(self.context, "kernel");

        let block_ids = opt.node_ids();
        let attributes = &[(
            Identifier::new(self.context, "llvm.emit_c_interface"),
            Attribute::unit(self.context).into(),
        )];

        let mut inputs = Vec::with_capacity(kernel.buffers.len());
        let mut block_input = Vec::with_capacity(kernel.buffers.len());

        for binding in kernel.buffers.iter() {
            let memref = self.item_to_memref_buffer_type(binding.item).into();
            inputs.push(memref);
            block_input.push((memref, self.location));
        }

        let func_type = TypeAttribute::new(FunctionType::new(self.context, &inputs, &[]).into());

        let location = self.location;
        module.body().append_operation(func::func(
            self.context,
            name,
            func_type,
            {
                let region = Region::new();
                let block = Block::new(&block_input);
                region.append_block(block);

                let block = region.first_block().unwrap();
                let argument_count = block.argument_count();
                for i in 0..argument_count {
                    self.global_buffers.push(block.argument(i).unwrap().into());
                }

                for basic_block_id in block_ids {
                    self.visit_basic_block(block, basic_block_id, opt);
                }

                block.append_operation(func::r#return(&[], location));

                self.block_stack.pop().unwrap();
                region
            },
            attributes,
            location,
        ));
    }
}
