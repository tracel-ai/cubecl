// use std::collections::HashMap;

use cubecl_core::ir::Builtin;

use super::prelude::*;

// enum MetadataKey {
//     Rank { var: Variable },
//     Stride { var: Variable },
//     Shape { var: Variable },
//     Length { var: Variable },
//     BufferLength { var: Variable },
// }

const NB_BUILTIN: usize = 30;

#[derive(Default)]
pub(super) struct ArgsManager<'a> {
    // memrefs: Vec<Value<'a, 'a>>,
    // metadata: HashMap<MetadataKey, Value<'a, 'a>>,
    builtin: [Option<Value<'a, 'a>>; NB_BUILTIN],
}

impl<'a> ArgsManager<'a> {
    // pub fn push_memref(&mut self, memref: Value<'a, 'a>) {
    // self.memrefs.push(memref);
    // }
    // pub fn get_memref(&self, number: usize) -> Value<'a, 'a> {
    // self.memrefs[number]
    // }
    pub fn set_builtin(&mut self, builtin: Builtin, value: Value<'a, 'a>) {
        self.builtin[builtin as usize] = Some(value);
    }
    pub fn get_builtin(&self, builtin: Builtin) -> Value<'a, 'a> {
        self.builtin[builtin as usize].expect("Unsupported builtin was used")
    }
}
