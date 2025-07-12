use std::collections::VecDeque;

use cubecl_core::{
    compute::{Binding, Location, Visibility},
    ir::{self, Id, VariableKind},
    prelude::KernelDefinition,
};
use cubecl_opt::{ConstArray, NodeIndex};
use hashbrown::{HashMap, HashSet};
use rspirv::spirv::{BuiltIn, CooperativeMatrixLayout, CooperativeMatrixUse, StorageClass, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::{ConstVal, Variable},
};

#[derive(Clone, Debug, Default)]
pub struct LookupTables {
    pub buffers: Vec<Word>,
    pub scalar_bindings: HashMap<ir::Elem, Word>,
    pub info: Word,
    pub cube_dims: Vec<Word>,
    pub cube_size: Word,

    pub const_arrays: Vec<Array>,
    pub shared_memories: HashMap<Id, Array>,
    pub local_arrays: HashMap<Id, Array>,
    pub matrices: HashMap<Id, Matrix>,

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub scalars: HashMap<(Id, ir::Elem), Word>,
    pub array_types: HashMap<Word, Word>,
    pub constants: HashMap<(ConstVal, Item), Word>,
    pub bindings: HashMap<Id, Word>,
    pub variables: HashMap<Id, Word>,
    pub versioned: HashMap<(Id, u16), Word>,
    pub labels: HashMap<NodeIndex, Word>,
    pub end_labels: HashMap<NodeIndex, Word>,

    pub slices: HashMap<Id, Slice>,

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
    pub var: ir::Variable,
    pub alignment: Option<u32>,
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

        self.state.buffers = kernel
            .buffers
            .into_iter()
            .map(|binding| {
                let var =
                    ir::Variable::new(VariableKind::GlobalInputArray(binding.id), binding.item);
                let name = self.name_of_var(var);
                target.generate_binding(self, binding, name.into())
            })
            .collect();

        let mut offset = self.state.buffers.len() as u32;
        let info_binding = Binding {
            id: offset,
            location: Location::Storage,
            visibility: Visibility::Read,
            item: ir::Item::new(ir::Elem::UInt(ir::UIntKind::U32)),
            size: None,
            has_extended_meta: false,
        };
        if self.metadata.static_len() > 0 {
            self.state.info = target.generate_binding(self, info_binding, "info".to_string());
            offset += 1;
        }

        self.state.scalar_bindings = kernel
            .scalars
            .into_iter()
            .enumerate()
            .map(|(i, binding)| {
                let elem = binding.elem;
                let binding = Binding {
                    id: i as u32 + offset,
                    location: Location::Storage,
                    visibility: Visibility::Read,
                    item: ir::Item::new(elem),
                    size: Some(binding.count),
                    has_extended_meta: false,
                };
                let name = format!("scalars({elem})");
                (elem, target.generate_binding(self, binding, name))
            })
            .collect();

        let cube_dims = [kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];
        self.state.cube_dims = cube_dims.iter().map(|dim| self.const_u32(*dim)).collect();
        self.state.cube_size = self.const_u32(cube_dims.iter().product());
    }

    pub fn const_u32(&mut self, value: u32) -> Word {
        let ty = Item::Scalar(Elem::Int(32, false));
        let ty_id = ty.id(self);
        self.constant_bit32(ty_id, value)
    }

    pub fn insert_global(&mut self, insert: impl FnOnce(&mut Self) -> Word) -> Word {
        let current_block = self.selected_block();
        let setup = self.setup_block;
        self.select_block(Some(setup)).unwrap();
        let id = insert(self);
        self.select_block(current_block).unwrap();
        id
    }

    pub fn get_local(&mut self, id: Id, item: &Item, var: ir::Variable) -> Word {
        if let Some(existing) = self.state.variables.get(&id) {
            *existing
        } else {
            let ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
            let word = self.declare_function_variable(ty);
            self.state.variables.insert(id, word);
            self.debug_var_name(word, var);
            word
        }
    }

    pub fn get_binding(&mut self, id: Id, var: &ir::Variable) -> Word {
        if let Some(existing) = self.state.bindings.get(&id) {
            *existing
        } else {
            let word = self.id();
            self.state.bindings.insert(id, word);
            self.debug_var_name(word, *var);
            word
        }
    }

    pub fn merge_binding(&mut self, id: Id, word: Word) {
        self.state.bindings.insert(id, word);
    }

    pub fn get_versioned(&mut self, id: (Id, u16), var: &ir::Variable) -> Word {
        if let Some(existing) = self.state.versioned.get(&id) {
            *existing
        } else {
            let word = self.id();
            self.state.versioned.insert(id, word);
            let mut debug_var = *var;
            debug_var.kind = VariableKind::LocalMut { id: id.0 };
            let name = self.name_of_var(debug_var);
            self.debug_name(word, format!("{name}.v{}", id.1));
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

    pub fn global_scalar(&mut self, id: Id, elem: ir::Elem) -> Variable {
        if let Some(existing) = self.state.scalars.get(&(id, elem)).copied() {
            let item = self.compile_item(ir::Item::new(elem));
            Variable::GlobalScalar(existing, item.elem())
        } else {
            let ir_var = ir::Variable::new(VariableKind::GlobalScalar(id), elem.into());
            let current_block = self.selected_block();
            let setup = self.setup_block;
            self.select_block(Some(setup)).unwrap();
            let arr_id = self.state.scalar_bindings[&elem];
            let item = self.compile_item(ir::Item::new(elem));
            let arr = Variable::GlobalInputArray(arr_id, item.clone(), 0);
            let const_id = self.const_u32(id);
            let index = Variable::ConstantScalar(const_id, id.into(), Elem::Int(32, false));
            let read_id = self.id();
            let var = Variable::GlobalScalar(read_id, item.elem());
            self.debug_var_name(read_id, ir_var);
            self.read_indexed_unchecked(&var, &arr, &index);
            self.select_block(current_block).unwrap();
            self.state.scalars.insert((id, elem), read_id);
            var
        }
    }

    pub fn register_const_array(&mut self, arr: ConstArray) {
        let var = ir::Variable::new(
            VariableKind::ConstantArray {
                id: arr.id,
                length: arr.length,
            },
            arr.item,
        );
        let item = self.compile_item(arr.item);
        let array_ty = Item::Array(Box::new(item.clone()), arr.length);
        let pointer_ty = Item::Pointer(StorageClass::Function, Box::new(array_ty.clone())).id(self);
        let array_ty = array_ty.id(self);
        let values = arr
            .values
            .into_iter()
            .map(|it| self.compile_variable(it))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|it| self.read_as(&it, &item))
            .collect::<Vec<_>>();
        let constant = self.constant_composite(array_ty, values);
        let id = self.variable(pointer_ty, None, StorageClass::Function, Some(constant));
        self.debug_var_name(id, var);
        self.state.const_arrays.insert(
            arr.id as usize,
            Array {
                id,
                item,
                len: arr.length,
                var,
                alignment: None,
            },
        );
    }
}
