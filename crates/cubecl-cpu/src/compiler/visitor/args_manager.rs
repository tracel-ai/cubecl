use cubecl_core::{Metadata, ir::Builtin, prelude::KernelDefinition};
use tracel_llvm::melior::ir::{
    Block, Location,
    r#type::{FunctionType, IntegerType, MemRefType},
};

use crate::compiler::builtin::BuiltinArray;

use super::prelude::*;

const NB_BUILTIN: usize = 30;

pub(super) struct ArgsManager<'a> {
    pub buffers: Vec<Value<'a, 'a>>,
    pub scalars: Vec<Value<'a, 'a>>,
    pub metadata_memref: Option<Value<'a, 'a>>,
    pub ext_meta_positions: Vec<u32>,
    pub metadata: Metadata,
    buffers_len: usize,
    scalars_len: usize,
    builtin: [Option<Value<'a, 'a>>; NB_BUILTIN],
    function_types: Vec<Type<'a>>,
    block_inputs: Vec<(Type<'a>, Location<'a>)>,
}

const NB_PASSED_BUILTIN: usize = 9;

impl<'a> ArgsManager<'a> {
    pub fn new(kernel: &KernelDefinition, context: &'a Context, location: Location<'a>) -> Self {
        let total_arg_len = kernel.buffers.len() + kernel.scalars.len() + NB_PASSED_BUILTIN;

        let mut num_ext = 0;
        let mut ext_meta_positions = vec![];

        let mut all_meta: Vec<_> = kernel
            .buffers
            .iter()
            .map(|buf| (buf.id, buf.has_extended_meta))
            .collect();

        all_meta.sort_by_key(|(id, _)| *id);

        for (_, has_extended_meta) in &all_meta {
            ext_meta_positions.push(num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        let num_meta = all_meta.len();

        let metadata = Metadata::new(num_meta as u32, num_ext);

        let mut args = ArgsManager {
            buffers: Vec::with_capacity(kernel.buffers.len()),
            buffers_len: kernel.buffers.len(),
            scalars: Vec::with_capacity(kernel.scalars.len()),
            scalars_len: kernel.scalars.len(),
            metadata_memref: None,
            builtin: [None; NB_BUILTIN],
            metadata,
            ext_meta_positions,
            function_types: Vec::with_capacity(total_arg_len),
            block_inputs: Vec::with_capacity(total_arg_len),
        };

        for binding in kernel.buffers.iter() {
            let inner_type = binding.item.elem.to_type(context);
            let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
            args.function_types.push(memref);
            args.block_inputs.push((memref, location));
        }

        // Metadata memref
        let inner_type = IntegerType::new(context, 32).into();
        let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
        args.function_types.push(memref);
        args.block_inputs.push((memref, location));

        for binding in kernel.scalars.iter() {
            let inner_type = binding.elem.to_type(context);
            let scalar = if binding.count > 1 {
                Type::vector(&[binding.count as u64], inner_type)
            } else {
                inner_type
            };
            args.function_types.push(scalar);
            args.block_inputs.push((scalar, location));
        }

        let integer_type: Type<'_> = IntegerType::new(context, 32).into();
        for _ in 0..9 {
            args.function_types.push(integer_type);
            args.block_inputs.push((integer_type, location));
        }

        args
    }

    pub fn ext_meta_position(&self, var: Variable) -> u32 {
        let id = var.index().expect("Variable should have index");
        self.ext_meta_positions[id as usize]
    }

    pub fn get_fn_type(&self, context: &'a Context) -> FunctionType<'a> {
        FunctionType::new(context, &self.function_types, &[])
    }

    pub fn create_top_block(&mut self) -> Block {
        let block = Block::new(&self.block_inputs);

        for i in 0..self.buffers_len {
            self.buffers.push(block.argument(i).unwrap().into());
        }

        self.metadata_memref = Some(block.argument(self.buffers_len).unwrap().into());

        for i in self.buffers_len + 1..self.buffers_len + 1 + self.scalars_len {
            self.scalars.push(block.argument(i).unwrap().into());
        }

        for (i, builtin) in BuiltinArray::builtin_order().into_iter().enumerate() {
            self.set_builtin(
                builtin,
                block
                    .argument(self.buffers_len + self.scalars_len + 1 + i)
                    .unwrap()
                    .into(),
            );
        }

        block
    }

    pub fn set_builtin(&mut self, builtin: Builtin, value: Value<'a, 'a>) {
        self.builtin[builtin as usize] = Some(value);
    }

    pub fn get_builtin(&self, builtin: Builtin) -> Value<'a, 'a> {
        self.builtin[builtin as usize]
            .expect(&format!("Unsupported builtin was used: {:?}", builtin))
    }
}
