use std::collections::HashMap;

use cubecl_core::{
    Metadata,
    ir::{Builtin, StorageType},
    prelude::{KernelDefinition, ScalarBinding},
};
use tracel_llvm::mlir_rs::ir::{
    Block, BlockRef, Location, Region,
    r#type::{FunctionType, IntegerType, MemRefType},
};

use crate::compiler::{
    builtin::BuiltinArray,
    passes::shared_memories::{SharedMemories, SharedMemory},
};

use super::prelude::*;

const NB_BUILTIN: usize = 31;

pub(super) struct ArgsManagerBuilder<'a, 'b> {
    scalars: Vec<ScalarBinding>,
    buffers_len: usize,
    function_types: Vec<Type<'a>>,
    metadata: Metadata,
    ext_meta_positions: Vec<u32>,
    block_inputs: Vec<(Type<'a>, Location<'a>)>,
    shared_memories: &'b SharedMemories,
    addr_type: Type<'a>,
    addr_size: usize,
}

impl<'a, 'b> ArgsManagerBuilder<'a, 'b> {
    pub fn new(
        kernel: &KernelDefinition,
        context: &'a Context,
        location: Location<'a>,
        shared_memories: &'b SharedMemories,
        addr_type: StorageType,
    ) -> Self {
        let total_arg_len = kernel.buffers.len()
            + kernel.scalars.len()
            + NB_PASSED_BUILTIN
            + shared_memories.0.len();

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

        let mut args = Self {
            buffers_len: kernel.buffers.len(),
            scalars,
            function_types: Vec::with_capacity(total_arg_len),
            block_inputs: Vec::with_capacity(total_arg_len),
            ext_meta_positions,
            shared_memories,
            metadata,
            addr_type: addr_type.to_type(context),
            addr_size: addr_type.size(),
        };

        for binding in kernel.buffers.iter() {
            let inner_type = binding.ty.storage_type().to_type(context);
            let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
            args.function_types.push(memref);
            args.block_inputs.push((memref, location));
        }

        for shared_memory in args.shared_memories.0.iter() {
            let memref = match shared_memory {
                SharedMemory::Array { ty, length, .. } => {
                    let inner_type = ty.to_type(context);
                    MemRefType::new(inner_type, &[*length as i64], None, None).into()
                }
                SharedMemory::Value { ty, .. } => {
                    let inner_type = ty.to_type(context);
                    MemRefType::new(inner_type, &[1], None, None).into()
                }
            };
            args.function_types.push(memref);
            args.block_inputs.push((memref, location));
        }

        // Metadata memref
        let inner_type = addr_type.to_type(context);
        let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
        args.function_types.push(memref);
        args.block_inputs.push((memref, location));

        for binding in kernel.scalars.iter() {
            let inner_type = binding.ty.to_type(context);
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

    pub fn get_fn_type(&self, context: &'a Context) -> FunctionType<'a> {
        FunctionType::new(context, &self.function_types, &[])
    }

    pub fn create_top_block(self, region: &Region<'a>) -> ArgsManager<'a> {
        let mut args = ArgsManager {
            buffers: Vec::with_capacity(self.buffers_len),
            scalars_memref: HashMap::with_capacity(self.scalars.len()),
            metadata_memref: None,
            builtin: [None; NB_BUILTIN],
            metadata: self.metadata.clone(),
            shared_memory_values: HashMap::with_capacity(self.shared_memories.0.len()),
            ext_meta_positions: self.ext_meta_positions.clone(),
            addr_type: self.addr_type,
            addr_size: self.addr_size,
        };

        let block = Block::new(&self.block_inputs);

        let mut total_len = 0;
        for i in 0..self.buffers_len {
            args.buffers.push(block.argument(i).unwrap().into());
        }

        total_len += self.buffers_len;

        for (i, shared_memory) in self.shared_memories.0.iter().enumerate() {
            let i = i + total_len;
            args.shared_memory_values
                .insert(shared_memory.id(), block.argument(i).unwrap().into());
        }

        total_len += self.shared_memories.0.len();

        args.metadata_memref = Some(block.argument(total_len).unwrap().into());
        total_len += 1;

        for i in 0..self.scalars.len() {
            let binding = &self.scalars[i];
            let i = i + total_len;
            args.scalars_memref
                .insert(binding.ty, block.argument(i).unwrap().into());
        }

        total_len += self.scalars.len();

        for (i, builtin) in BuiltinArray::builtin_order().into_iter().enumerate() {
            let i = i + total_len;
            args.set(builtin, block.argument(i).unwrap().into());
        }

        region.append_block(block);
        args
    }
}

pub(super) struct ArgsManager<'a> {
    pub buffers: Vec<Value<'a, 'a>>,
    pub scalars_memref: HashMap<StorageType, Value<'a, 'a>>,
    pub metadata_memref: Option<Value<'a, 'a>>,
    pub ext_meta_positions: Vec<u32>,
    pub metadata: Metadata,
    pub shared_memory_values: HashMap<u32, Value<'a, 'a>>,
    pub builtin: [Option<Value<'a, 'a>>; NB_BUILTIN],
    pub addr_type: Type<'a>,
    pub addr_size: usize,
}

const NB_PASSED_BUILTIN: usize = 9;

impl<'a> ArgsManager<'a> {
    pub fn buffer_position(&self, var: Variable) -> u32 {
        var.index().expect("Variable should have index")
    }

    pub fn ext_meta_position(&self, var: Variable) -> u32 {
        let id = var.index().expect("Variable should have index");
        self.ext_meta_positions[id as usize]
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

        let unit_pos_yz_corrected = block
            .addi(unit_pos_z_corrected, unit_pos_y_corrected, location)
            .unwrap();
        let unit_pos = block
            .addi(unit_pos_yz_corrected, self.get(Builtin::UnitPosX), location)
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

    pub fn as_address_type<'b, 'c: 'b>(
        &self,
        value: Value<'c, 'c>,
        block: &'b Block<'c>,
        location: Location<'c>,
    ) -> Value<'c, 'c>
    where
        'a: 'c,
    {
        if self.addr_size > 4 {
            // Lifetime for this function is inconsistent with other arithmetic, so need to cast the
            // output type to make it work.
            let value: Value<'c, 'b> = block.extui(value, self.addr_type, location).unwrap();
            unsafe { core::mem::transmute::<Value<'c, 'b>, Value<'c, 'c>>(value) }
        } else {
            value
        }
    }
}
