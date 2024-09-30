use std::collections::VecDeque;

use cubecl_core::ir::KernelDefinition;
use hashbrown::HashMap;
use rspirv::spirv::{BuiltIn, Word};

use crate::{
    containers::{Array, Slice},
    item::{Elem, Item},
    variable::Globals,
    SpirvCompiler, SpirvTarget,
};

#[derive(Clone, Debug, Default)]
pub struct LookupTables {
    pub inputs: Vec<Word>,
    pub outputs: Vec<Word>,
    pub named: HashMap<String, Word>,
    pub cube_dims: Vec<Word>,
    pub cube_size: Word,

    pub const_arrays: Vec<Array>,
    pub shared_memories: Vec<Array>,
    pub local_arrays: HashMap<(u16, u8), Array>,

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub globals: HashMap<Globals, Word>,
    pub array_types: HashMap<Word, Word>,
    pub constants: HashMap<(u64, Item), Word>,
    pub bindings: HashMap<(u16, u8), Word>,
    pub variables: HashMap<(u16, u8), Word>,

    pub slices: HashMap<(u16, u8), Slice>,

    pub rank: Word,
    pub extensions: Vec<Word>,
    // For break, continue
    pub loops: VecDeque<Loop>,
}

#[derive(Clone, Debug)]
pub struct Loop {
    pub header: Word,
    pub continue_target: Word,
    pub post: Word,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_state(&mut self, kernel: KernelDefinition) {
        let mut target = self.target.clone();

        self.state.inputs = kernel
            .inputs
            .into_iter()
            .enumerate()
            .map(|(i, binding)| target.generate_binding(self, binding, i as u32))
            .collect();
        let offset = self.state.inputs.len() as u32;
        self.state.outputs = kernel
            .outputs
            .into_iter()
            .enumerate()
            .map(|(i, binding)| target.generate_binding(self, binding, i as u32 + offset))
            .collect();
        let offset = offset + self.state.outputs.len() as u32;
        self.state.named = kernel
            .named
            .into_iter()
            .enumerate()
            .map(|(i, (name, binding))| {
                (
                    name,
                    target.generate_binding(self, binding, i as u32 + offset),
                )
            })
            .collect();

        let cube_dims = [kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];
        self.state.cube_dims = cube_dims.iter().map(|dim| self.const_u32(*dim)).collect();
        self.state.cube_size = self.const_u32(cube_dims.iter().product());
    }

    pub fn const_u32(&mut self, value: u32) -> Word {
        let ty = Item::Scalar(Elem::Int(32, false));
        let ty_id = ty.id(self);
        self.get_or_insert_const(value as u64, ty, |b| b.constant_bit32(ty_id, value))
    }

    pub fn get_or_insert_const(
        &mut self,
        value: u64,
        item: Item,
        insert: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        if let Some(id) = self.state.constants.get(&(value, item.clone())) {
            *id
        } else {
            let id = insert(self);
            self.state.constants.insert((value, item), id);
            id
        }
    }

    pub fn get_or_insert_global(
        &mut self,
        global: Globals,
        insert: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        if let Some(id) = self.state.globals.get(&global) {
            *id
        } else {
            let id = insert(self);
            self.state.globals.insert(global, id);
            id
        }
    }
}
