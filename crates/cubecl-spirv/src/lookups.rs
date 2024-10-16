use std::collections::VecDeque;

use cubecl_core::ir::{self, KernelDefinition};
use cubecl_opt::NodeIndex;
use hashbrown::{HashMap, HashSet};
use rspirv::spirv::{BuiltIn, CooperativeMatrixLayout, CooperativeMatrixUse, StorageClass, Word};

use crate::{
    item::{Elem, Item},
    variable::{ConstVal, Globals, Variable},
    SpirvCompiler, SpirvTarget,
};

#[derive(Clone, Debug, Default)]
pub struct LookupTables {
    pub inputs: Vec<Word>,
    pub outputs: Vec<Word>,
    pub named: HashMap<String, Word>,
    pub cube_dims: Vec<Word>,
    pub cube_size: Word,

    pub const_arrays: Vec<ConstArray>,
    pub shared_memories: HashMap<u16, Array>,
    pub local_arrays: HashMap<(u16, u8), Array>,
    pub matrices: HashMap<(u16, u8), Matrix>,

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub scalars: HashMap<(u16, ir::Elem), Word>,
    pub globals: HashMap<Globals, Word>,
    pub array_types: HashMap<Word, Word>,
    pub constants: HashMap<(ConstVal, Item), Word>,
    pub bindings: HashMap<(u16, u8), Word>,
    pub variables: HashMap<(u16, u8), Word>,
    pub versioned: HashMap<(u16, u8, u16), Word>,
    pub labels: HashMap<NodeIndex, Word>,
    pub end_labels: HashMap<NodeIndex, Word>,

    pub slices: HashMap<(u16, u8), Slice>,

    pub extensions: Vec<Word>,
    // For break, continue
    pub loops: VecDeque<Loop>,

    pub debug_types: HashSet<Word>,
}

#[derive(Clone, Debug)]
pub struct Slice {
    pub ptr: Variable,
    pub offset: Word,
    pub end: Word,
    pub const_len: Option<u32>,
    pub item: Item,
}

impl From<&Slice> for Variable {
    fn from(value: &Slice) -> Self {
        Variable::Slice {
            ptr: Box::new(value.ptr.clone()),
            offset: value.offset,
            end: value.end,
            const_len: value.const_len,
            item: value.item.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Array {
    pub id: Word,
    pub item: Item,
    pub len: u32,
}

#[derive(Clone, Debug)]
pub struct ConstArray {
    pub id: Word,
    pub item: Item,
    pub len: u32,
    pub composite_id: Word,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix {
    pub id: Word,
    pub ident: CooperativeMatrixUse,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub elem: Elem,
    pub layout: Option<CooperativeMatrixLayout>,
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
            .map(|(i, binding)| {
                target.generate_binding(self, binding, format!("input({i})"), i as u32)
            })
            .collect();
        let offset = self.state.inputs.len() as u32;
        self.state.outputs = kernel
            .outputs
            .into_iter()
            .enumerate()
            .map(|(i, binding)| {
                target.generate_binding(self, binding, format!("output({i})"), i as u32 + offset)
            })
            .collect();
        let offset = offset + self.state.outputs.len() as u32;
        self.state.named = kernel
            .named
            .into_iter()
            .enumerate()
            .map(|(i, (name, binding))| {
                (
                    name.clone(),
                    target.generate_binding(self, binding, name, i as u32 + offset),
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
        self.get_or_insert_const(ConstVal::Bit32(value), ty, |b| {
            b.constant_bit32(ty_id, value)
        })
    }

    pub fn get_or_insert_const(
        &mut self,
        value: ConstVal,
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
            let current_block = self.selected_block();
            let setup = self.setup_block;
            self.select_block(Some(setup)).unwrap();
            let id = insert(self);
            self.select_block(current_block).unwrap();
            self.state.globals.insert(global, id);
            id
        }
    }

    pub fn get_local(&mut self, id: (u16, u8), item: &Item) -> Word {
        if let Some(existing) = self.state.variables.get(&id) {
            *existing
        } else {
            let ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
            let word = self.declare_function_variable(ty);
            self.state.variables.insert(id, word);
            self.debug_name(word, format!("local({}, {})", id.0, id.1));
            word
        }
    }

    pub fn get_binding(&mut self, id: (u16, u8)) -> Word {
        if let Some(existing) = self.state.bindings.get(&id) {
            *existing
        } else {
            let word = self.id();
            self.state.bindings.insert(id, word);
            self.debug_name(word, format!("binding({}, {})", id.0, id.1));
            word
        }
    }

    pub fn merge_binding(&mut self, id: (u16, u8), word: Word) {
        self.state.bindings.insert(id, word);
    }

    pub fn get_versioned(&mut self, id: (u16, u8, u16)) -> Word {
        if let Some(existing) = self.state.versioned.get(&id) {
            *existing
        } else {
            let word = self.id();
            self.state.versioned.insert(id, word);
            self.debug_name(word, format!("local({}, {}).v{}", id.0, id.1, id.2));
            word
        }
    }

    pub fn label(&mut self, block: NodeIndex) -> Word {
        if let Some(existing) = self.state.labels.get(&block) {
            *existing
        } else {
            let word = self.id();
            self.debug_name(word, format!("bb{}", block.index()));
            self.state.labels.insert(block, word);
            word
        }
    }

    pub fn end_label(&mut self, block: NodeIndex) -> Word {
        if let Some(existing) = self.state.end_labels.get(&block) {
            *existing
        } else {
            let word = self.label(block);
            self.state.end_labels.insert(block, word);
            word
        }
    }

    pub fn global_scalar(&mut self, id: u16, elem: ir::Elem) -> Variable {
        if let Some(existing) = self.state.scalars.get(&(id, elem)).copied() {
            let item = self.compile_item(ir::Item::new(elem));
            Variable::GlobalScalar(existing, item.elem())
        } else {
            let current_block = self.selected_block();
            let setup = self.setup_block;
            self.select_block(Some(setup)).unwrap();
            let arr_id = self.state.named[&format!("scalars_{elem}")];
            let item = self.compile_item(ir::Item::new(elem));
            let arr = Variable::GlobalInputArray(arr_id, item.clone());
            let const_id = self.const_u32(id as u32);
            let index =
                Variable::ConstantScalar(const_id, (id as u32).into(), Elem::Int(32, false));
            let read_id = self.id();
            let var = Variable::GlobalScalar(read_id, item.elem());
            self.debug_name(read_id, format!("scalars<{elem}>({id})"));
            self.read_indexed_unchecked(&var, &arr, &index);
            self.select_block(current_block).unwrap();
            self.state.scalars.insert((id, elem), read_id);
            var
        }
    }
}
