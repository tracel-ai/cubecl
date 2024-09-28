use cubecl_core::ir::KernelDefinition;
use hashbrown::HashMap;
use rspirv::spirv::{BuiltIn, Word};

use crate::{
    containers::Slice,
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

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    // Need separate tracking so we can decorate strides
    pub array_types: HashMap<Word, Word>,
    pub globals: HashMap<Globals, Word>,

    pub constants: HashMap<(u64, Elem), Word>,
    pub bindings: HashMap<(u16, u8), Word>,
    pub variables: HashMap<(u16, u8), Word>,

    pub slices: HashMap<(u16, u8), Slice>,

    pub rank: Word,
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
    }

    pub fn const_u32(&mut self, value: u32) -> Word {
        let ty = Elem::Int(32);
        let ty_id = ty.id(self);
        self.get_or_insert_const(value as u64, ty, |b| b.constant_bit32(ty_id, value))
    }

    pub fn get_or_insert_const(
        &mut self,
        value: u64,
        elem: Elem,
        insert: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        if let Some(id) = self.state.constants.get(&(value, elem)) {
            *id
        } else {
            let id = insert(self);
            self.state.constants.insert((value, elem), id);
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
