use std::collections::VecDeque;

use cubecl_core::{
    ir::{self, Id, Type, VariableKind},
    prelude::{Binding, KernelDefinition, Location, Visibility},
};
use cubecl_opt::{ConstArray, NodeIndex, SharedMemory};
use hashbrown::{HashMap, HashSet};
use rspirv::{
    dr,
    spirv::{self, BuiltIn, CooperativeMatrixLayout, CooperativeMatrixUse, StorageClass, Word},
};

use crate::{
    MAX_VECTORIZATION, SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::{ConstVal, Variable},
};

#[derive(Clone, Debug, Default)]
pub struct LookupTables {
    pub buffers: Vec<Word>,
    pub scalar_bindings: HashMap<ir::StorageType, Word>,
    pub info: Word,
    pub cube_dims: Vec<Word>,
    pub cube_size: Word,

    pub const_arrays: Vec<Array>,
    pub shared_arrays: HashMap<Id, SharedArray>,
    pub shared: HashMap<Id, SharedVar>,
    pub local_arrays: HashMap<Id, Array>,
    pub matrices: HashMap<Id, Matrix>,

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub scalars: HashMap<(Id, ir::StorageType), Word>,
    pub array_types: HashSet<Word>,
    pub constants: HashMap<(ConstVal, Item), Word>,
    pub bindings: HashMap<Id, Word>,
    pub variables: HashMap<Id, Word>,
    pub versioned: HashMap<(Id, u16), Word>,
    pub labels: HashMap<NodeIndex, Word>,
    pub end_labels: HashMap<NodeIndex, Word>,

    pub slices: HashMap<Id, Slice>,

    // For break, continue
    pub loops: VecDeque<Loop>,

    // Explicitly decorated types, to avoid double decorating
    pub decorated_types: HashSet<Word>,
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

#[derive(Clone, Debug)]
pub struct SharedArray {
    pub id: Word,
    pub item: Item,
    pub len: u32,
    pub align: u32,
    pub offset: u32,
}

#[derive(Clone, Debug)]
pub struct SharedVar {
    pub id: Word,
    pub item: Item,
    pub offset: u32,
    pub align: u32,
}

impl SharedArray {
    pub fn end(&self) -> u32 {
        self.offset + self.len * self.item.size()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix {
    pub id: Word,
    pub ident: CooperativeMatrixUse,
    pub m: u32,
    pub n: u32,
    pub k: u32,
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
            .map(|mut binding| {
                // This is safe when combined with the unroll transform that adjusts all indices.
                // Must not be used alone
                if binding.ty.line_size() > MAX_VECTORIZATION {
                    binding.ty = binding.ty.line(MAX_VECTORIZATION);
                }
                let var = ir::Variable::new(VariableKind::GlobalInputArray(binding.id), binding.ty);
                let name = self.name_of_var(var);
                target.generate_binding(self, binding, name.into())
            })
            .collect();

        let mut offset = self.state.buffers.len() as u32;
        let info_binding = Binding {
            id: offset,
            location: Location::Storage,
            visibility: Visibility::Read,
            ty: self.addr_type.into(),
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
                let elem = binding.ty;
                let binding = Binding {
                    id: i as u32 + offset,
                    location: Location::Storage,
                    visibility: Visibility::Read,
                    ty: ir::Type::new(elem),
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

        let shared_liveness = self.shared_liveness.clone();
        for alloc in shared_liveness.allocations.values() {
            let smem_id = self.id();

            match alloc.smem {
                SharedMemory::Array {
                    id,
                    length,
                    ty,
                    align,
                } => {
                    let item = self.compile_type(ty);
                    self.state.shared_arrays.insert(
                        id,
                        SharedArray {
                            id: smem_id,
                            item,
                            len: length as u32,
                            align: align as u32,
                            offset: alloc.offset as u32,
                        },
                    );
                }
                SharedMemory::Value { id, ty, align } => {
                    let item = self.compile_type(ty);
                    self.state.shared.insert(
                        id,
                        SharedVar {
                            id: smem_id,
                            item,
                            offset: alloc.offset as u32,
                            align: align as u32,
                        },
                    );
                }
            }
        }
    }

    fn dedup_const(&mut self, inst: &dr::Instruction) -> Option<Word> {
        self.module_ref()
            .types_global_values
            .iter()
            .find(|it| {
                it.class == inst.class
                    && it.result_type == inst.result_type
                    && it.operands == inst.operands
            })
            .and_then(|it| it.result_id)
    }

    pub fn dedup_constant_bit32(&mut self, ty: Word, val: u32) -> Word {
        let inst = dr::Instruction::new(
            spirv::Op::Constant,
            Some(ty),
            None,
            vec![dr::Operand::LiteralBit32(val)],
        );
        if let Some(id) = self.dedup_const(&inst) {
            id
        } else {
            self.constant_bit32(ty, val)
        }
    }

    pub fn dedup_constant_bit64(&mut self, ty: Word, val: u64) -> Word {
        let inst = dr::Instruction::new(
            spirv::Op::Constant,
            Some(ty),
            None,
            vec![dr::Operand::LiteralBit64(val)],
        );
        if let Some(id) = self.dedup_const(&inst) {
            id
        } else {
            self.constant_bit64(ty, val)
        }
    }

    pub fn const_u32(&mut self, value: u32) -> Word {
        let ty = Item::Scalar(Elem::Int(32, false));
        let ty_id = ty.id(self);
        self.dedup_constant_bit32(ty_id, value)
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

    pub fn global_scalar(&mut self, id: Id, ty: ir::StorageType) -> Variable {
        if let Some(existing) = self.state.scalars.get(&(id, ty)).copied() {
            let item = self.compile_type(ir::Type::new(ty));
            Variable::GlobalScalar(existing, item.elem())
        } else {
            let ir_var = ir::Variable::new(VariableKind::GlobalScalar(id), Type::new(ty));
            let current_block = self.selected_block();
            let setup = self.setup_block;
            self.select_block(Some(setup)).unwrap();
            let arr_id = self.state.scalar_bindings[&ty];
            let item = self.compile_type(ir::Type::new(ty));
            let arr = Variable::GlobalInputArray(arr_id, item.clone(), 0);
            let const_id = self.const_u32(id);
            let index = Variable::ConstantScalar(const_id, id.into(), Elem::Int(32, false));
            let read_id = self.id();
            let var = Variable::GlobalScalar(read_id, item.elem());
            self.debug_var_name(read_id, ir_var);
            self.read_indexed_unchecked(&var, &arr, &index);
            self.select_block(current_block).unwrap();
            self.state.scalars.insert((id, ty), read_id);
            var
        }
    }

    pub fn register_const_array(&mut self, arr: ConstArray) {
        let var = ir::Variable::new(
            VariableKind::ConstantArray {
                id: arr.id,
                length: arr.length,
                unroll_factor: 1,
            },
            arr.item,
        );
        let item = self.compile_type(arr.item);
        let array_ty = Item::Array(Box::new(item.clone()), arr.length as u32);
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
                len: arr.length as u32,
                var,
                alignment: None,
            },
        );
    }
}
