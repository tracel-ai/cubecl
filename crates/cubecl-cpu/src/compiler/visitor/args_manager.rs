use std::collections::HashMap;

use cubecl_core::{
    Metadata,
    compute::ScalarBinding,
    ir::{Builtin, Elem},
    prelude::KernelDefinition,
};
use tracel_llvm::melior::ir::{
    Block, BlockRef, Location,
    r#type::{FunctionType, IntegerType, MemRefType},
};

use crate::compiler::builtin::BuiltinArray;

use super::prelude::*;

const NB_BUILTIN: usize = 30;

pub(super) struct ArgsManager<'a> {
    pub buffers: Vec<Value<'a, 'a>>,
    pub scalars_memref: HashMap<Elem, Value<'a, 'a>>,
    pub metadata_memref: Option<Value<'a, 'a>>,
    pub ext_meta_positions: Vec<u32>,
    pub metadata: Metadata,
    scalars: Vec<ScalarBinding>,
    buffers_len: usize,
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
        let scalars = kernel.scalars.clone();

        let mut args = ArgsManager {
            buffers: Vec::with_capacity(kernel.buffers.len()),
            buffers_len: kernel.buffers.len(),
            scalars_memref: HashMap::with_capacity(kernel.scalars.len()),
            scalars,
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
            let memref = MemRefType::new(inner_type, &[binding.count as i64], None, None).into();
            args.function_types.push(memref);
            args.block_inputs.push((memref, location));
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

        for i in 0..self.scalars.len() {
            let binding = &self.scalars[i];
            self.scalars_memref.insert(
                binding.elem,
                block.argument(i + self.buffers_len + 1).unwrap().into(),
            );
        }

        for (i, builtin) in BuiltinArray::builtin_order().into_iter().enumerate() {
            self.set(
                builtin,
                block
                    .argument(self.buffers_len + self.scalars.len() + 1 + i)
                    .unwrap()
                    .into(),
            );
        }

        block
    }

    pub fn compute_derived_args_builtin(
        &mut self,
        block: BlockRef<'a, 'a>,
        location: Location<'a>,
    ) {
        let cube_dim_xy = block
            .muli(
                self.get(Builtin::CubeDimX),
                self.get(Builtin::CubeDimY),
                location,
            )
            .unwrap();
        let cube_dim = block
            .muli(cube_dim_xy, self.get(Builtin::CubeDimZ), location)
            .unwrap();
        self.set(Builtin::CubeDim, cube_dim);

        let unit_pos_z_corrected = block
            .muli(self.get(Builtin::UnitPosZ), cube_dim_xy, location)
            .unwrap();

        let unit_pos_y_corrected = block
            .muli(
                self.get(Builtin::UnitPosY),
                self.get(Builtin::CubeDimX),
                location,
            )
            .unwrap();

        let unit_pos_xy_corrected = block
            .addi(unit_pos_z_corrected, unit_pos_y_corrected, location)
            .unwrap();
        let unit_pos = block
            .addi(unit_pos_xy_corrected, self.get(Builtin::UnitPosX), location)
            .unwrap();
        self.set(Builtin::UnitPos, unit_pos);
    }

    pub fn set(&mut self, builtin: Builtin, value: Value<'a, 'a>) {
        self.builtin[builtin as usize] = Some(value);
    }

    pub fn get(&self, builtin: Builtin) -> Value<'a, 'a> {
        self.builtin[builtin as usize]
            .unwrap_or_else(|| panic!("Unsupported builtin was used: {builtin:?}"))
    }
}
