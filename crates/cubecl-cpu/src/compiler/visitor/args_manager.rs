// use std::collections::HashMap;

use cubecl_core::{
    compute::{Binding, ScalarBinding},
    ir::Builtin,
};
use tracel_llvm::melior::ir::{
    Block, Location,
    r#type::{FunctionType, IntegerType, MemRefType},
};

use crate::compiler::builtin::BuiltinArray;

use super::prelude::*;

// enum MetadataKey {
//     Rank { var: Variable },
//     Stride { var: Variable },
//     Shape { var: Variable },
//     Length { var: Variable },
//     BufferLength { var: Variable },
// }

const NB_BUILTIN: usize = 30;

pub(super) struct ArgsManager<'a> {
    pub buffers: Vec<Value<'a, 'a>>,
    pub scalars: Vec<Value<'a, 'a>>,
    buffers_len: usize,
    scalars_len: usize,
    // metadata: HashMap<MetadataKey, Value<'a, 'a>>,
    builtin: [Option<Value<'a, 'a>>; NB_BUILTIN],
    function_types: Vec<Type<'a>>,
    block_inputs: Vec<(Type<'a>, Location<'a>)>,
}

const NB_PASSED_BUILTIN: usize = 9;

impl<'a> ArgsManager<'a> {
    pub fn new(
        bindings: &[Binding],
        scalar_bindings: &[ScalarBinding],
        context: &'a Context,
        location: Location<'a>,
    ) -> Self {
        let total_arg_len = bindings.len() + scalar_bindings.len() + NB_PASSED_BUILTIN;
        let mut args = ArgsManager {
            buffers: Vec::with_capacity(bindings.len()),
            buffers_len: bindings.len(),
            scalars: Vec::with_capacity(scalar_bindings.len()),
            scalars_len: scalar_bindings.len(),
            builtin: [None; NB_BUILTIN],
            function_types: Vec::with_capacity(total_arg_len),
            block_inputs: Vec::with_capacity(total_arg_len),
        };

        for binding in bindings.iter() {
            let inner_type = binding.item.elem.to_type(context);
            let memref = MemRefType::new(inner_type, &[i64::MIN], None, None).into();
            args.function_types.push(memref);
            args.block_inputs.push((memref, location));
        }

        for binding in scalar_bindings.iter() {
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

    pub fn get_fn_type(&self, context: &'a Context) -> FunctionType<'a> {
        FunctionType::new(context, &self.function_types, &[])
    }

    pub fn create_top_block(&mut self) -> Block {
        let block = Block::new(&self.block_inputs);

        for i in 0..self.buffers_len {
            self.buffers.push(block.argument(i).unwrap().into());
        }

        for i in self.buffers_len..self.buffers_len + self.scalars_len {
            self.scalars.push(block.argument(i).unwrap().into());
        }

        for (i, builtin) in BuiltinArray::builtin_order().into_iter().enumerate() {
            self.set_builtin(
                builtin,
                block
                    .argument(self.buffers_len + self.scalars_len + i)
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
        self.builtin[builtin as usize].expect("Unsupported builtin was used")
    }
}
