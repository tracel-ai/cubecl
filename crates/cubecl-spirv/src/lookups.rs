use core::ops::{Deref, DerefMut};
use std::collections::VecDeque;

use cubecl_core::{
    ir::{self, Builtin, Id, Type, VariableKind},
    prelude::KernelDefinition,
};
use cubecl_opt::{ConstArray, NodeIndex};
use hashbrown::{HashMap, HashSet};
use rspirv::{
    dr,
    spirv::{
        self, BuiltIn, CooperativeMatrixLayout, CooperativeMatrixUse, MemoryAccess, Scope,
        StorageClass, Word,
    },
};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::{ConstVal, Variable},
};

#[derive(Clone, Debug, Default)]
pub struct CompilerState {
    pub extra_funcs: HashMap<Id, FuncDefinition>,

    pub scalar_bindings: HashMap<ir::StorageType, u32>,
    pub params: Word,
    pub info: Option<Buffer>,
    pub cube_dims: Vec<Word>,
    pub cube_size: Word,

    // For break, continue
    pub loops: VecDeque<Loop>,

    pub debug_types: HashSet<Word>,

    /// Base lookups, used to query global parameters like buffers and shared memory for function
    /// definitions
    pub base_lookups: LookupTables,
    /// Lookups for the current function
    pub lookups: LookupTables,
}

impl Deref for CompilerState {
    type Target = LookupTables;

    fn deref(&self) -> &Self::Target {
        &self.lookups
    }
}

impl DerefMut for CompilerState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lookups
    }
}

#[derive(Clone, Debug, Default)]
pub struct LookupTables {
    pub buffers: Vec<Buffer>,

    pub const_arrays: Vec<Array>,
    pub shared: HashMap<Id, SharedVar>,

    pub matrices: HashMap<Id, Matrix>,
    pub globals: HashMap<Builtin, Word>,
    pub loaded_builtins: HashMap<BuiltIn, Word>,

    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub scalars: HashMap<(Id, ir::StorageType), Word>,
    pub constants: HashMap<(ConstVal, Item), Word>,
    pub bindings: HashMap<Id, Word>,
    pub variables: HashMap<Id, Word>,
    pub versioned: HashMap<(Id, u16), Word>,
    pub labels: HashMap<NodeIndex, Word>,
    pub end_labels: HashMap<NodeIndex, Word>,

    pub atomic_scopes: HashMap<Word, Scope>,

    pub slices: HashMap<Id, Slice>,
}

#[derive(Clone, Debug)]
pub struct Slice {
    pub ptr: Variable,
    pub offset: Word,
    pub end: Word,
    pub const_len: Option<u32>,
    pub item: Item,
}

#[derive(Clone, Debug)]
pub struct FuncDefinition {
    pub type_id: Word,
    pub id: Word,
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
pub struct SharedVar {
    pub id: Word,
    pub ptr_ty_id: Word,
    pub item: Item,
    pub offset: u32,
    pub align: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix {
    pub id: Word,
    pub ident: CooperativeMatrixUse,
    pub scope: Scope,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Buffer {
    pub id: Word,
    pub struct_ty_id: Word,
    pub struct_ptr_ty_id: Word,
    pub storage_class: StorageClass,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn init_base_state(&mut self, kernel: &mut KernelDefinition) {
        let cube_dims = [kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z];
        self.state.cube_dims = cube_dims.iter().map(|dim| self.const_u32(*dim)).collect();
        self.state.cube_size = self.const_u32(cube_dims.iter().product());

        let mut target = self.target.clone();

        let max_vector_size = self.compilation_options.vulkan.max_vector_size;
        for binding in &mut kernel.buffers {
            // This is safe when combined with the unroll transform that adjusts all indices.
            // Must not be used alone
            if binding.ty.vector_size() > max_vector_size {
                binding.ty = binding.ty.with_vector_size(max_vector_size);
            }
        }

        let opt = self.opt.clone();
        let mut visibility = opt.global_state.buffer_visibility.borrow_mut();
        // Just in case not all buffers were accessed when tracking reads/writes
        visibility.resize(kernel.num_global_buffers(), Default::default());
        self.state.base_lookups.buffers =
            target.generate_params(self, &kernel.buffers, &visibility);

        let shared_liveness = self.shared_liveness.clone();
        for alloc in shared_liveness.allocations.values() {
            let smem_id = self.id();
            let smem_ptr_ty_id = self.id();

            let item = self.compile_type(alloc.smem.ty);
            self.state.base_lookups.shared.insert(
                alloc.smem.id,
                SharedVar {
                    id: smem_id,
                    ptr_ty_id: smem_ptr_ty_id,
                    item,
                    offset: alloc.offset as u32,
                    align: alloc.smem.align as u32,
                },
            );
        }

        self.state.lookups = self.state.base_lookups.clone();
    }

    pub fn init_kernel_state(&mut self, kernel: KernelDefinition) {
        self.state.scalar_bindings = kernel
            .scalars
            .into_iter()
            .enumerate()
            .map(|(i, arg)| (arg.ty, i as u32))
            .collect();
        self.state.lookups.buffers = self.state.base_lookups.buffers.clone();
        self.state.lookups.shared = self.state.base_lookups.shared.clone();
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

    pub fn insert_builtin(
        &mut self,
        builtin: BuiltIn,
        insert: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        if let Some(id) = self.state.loaded_builtins.get(&builtin) {
            *id
        } else {
            let id = self.insert_in_setup(insert);
            self.state.loaded_builtins.insert(builtin, id);
            id
        }
    }

    pub fn insert_global(
        &mut self,
        builtin: Builtin,
        insert: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        if let Some(id) = self.state.globals.get(&builtin) {
            *id
        } else {
            let id = self.insert_in_setup(insert);
            self.state.globals.insert(builtin, id);
            id
        }
    }

    pub fn insert_in_setup(&mut self, insert: impl FnOnce(&mut Self) -> Word) -> Word {
        let current_block = self.selected_block();
        let setup = self.setup_block;
        self.select_block(Some(setup)).unwrap();
        let id = insert(self);
        self.select_block(current_block).unwrap();
        id
    }

    pub fn insert_in_root(&mut self, insert: impl FnOnce(&mut Self) -> Word) -> Word {
        let current_block = self.selected_block();
        self.select_block(None).unwrap();
        let id = insert(self);
        self.select_block(current_block).unwrap();
        id
    }

    pub fn get_local(&mut self, id: Id, item: &Item, var: ir::Variable) -> Word {
        if let Some(existing) = self.state.variables.get(&id) {
            *existing
        } else {
            let ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
            let word = self.declare_function_variable(ty, None);
            self.state.variables.insert(id, word);
            self.debug_var_name(word, var);
            word
        }
    }

    fn init_local_from_param(&mut self, id: Id, item: &Item, var: ir::Variable, init: Word) {
        let ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
        let word = self.declare_function_variable(ty, Some(init));
        self.state.variables.insert(id, word);
        self.debug_var_name(word, var);
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

    pub fn init_function_param(&mut self, param: ir::Variable, param_id: Word) {
        let item = self.compile_type(param.ty);
        match param.kind {
            VariableKind::GlobalBuffer(id) | VariableKind::TensorMap(id) => {
                self.state.buffers[id as usize].id = param_id;
            }
            VariableKind::Shared { id, .. } => {
                self.state.shared.get_mut(&id).unwrap().id = param_id;
            }
            VariableKind::Builtin(builtin) => {
                self.state.globals.insert(builtin, param_id);
            }
            VariableKind::ConstantArray { .. }
            | VariableKind::Pipeline { .. }
            | VariableKind::BarrierToken { .. } => {
                panic!("{param} not allowed as a function param")
            }
            VariableKind::Constant(value) => {
                let const_val = (value, item.clone()).into();
                self.state.constants.insert((const_val, item), param_id);
            }
            VariableKind::GlobalScalar(id) => {
                self.state
                    .scalars
                    .insert((id, param.storage_type()), param_id);
            }
            VariableKind::LocalMut { id } => self.init_local_from_param(id, &item, param, param_id),
            VariableKind::LocalConst { id } => {
                self.state.bindings.insert(id, param_id);
            }
            VariableKind::Versioned { id, version } => {
                self.state.versioned.insert((id, version), param_id);
            }
            VariableKind::Matrix { id, mat } => {
                let matrix = self.init_coop_matrix(mat, param, Some(param_id));
                self.state.matrices.insert(id, matrix);
            }
            VariableKind::Aggregate { .. } => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    pub fn end_function_and_reset_lookups(&mut self) {
        self.builder.end_function().unwrap();
        self.state.lookups = self.state.base_lookups.clone();
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
            let field_id = self.const_u32(self.state.scalar_bindings[&ty]);
            let offset = self.const_u32(id);
            let item = self.compile_type(ir::Type::new(ty));
            let align = item.size();
            let elem = item.elem();
            let ty_id = item.id(self);
            let storage_class = T::info_storage_class(self);
            let ptr_ty = Item::Pointer(storage_class, Box::new(item)).id(self);
            let info = self.state.info.unwrap().id;
            let access = self
                .in_bounds_access_chain(ptr_ty, None, info, [field_id, offset])
                .unwrap();
            let read_id = self
                .load(
                    ty_id,
                    None,
                    access,
                    Some(MemoryAccess::ALIGNED),
                    [align.into()],
                )
                .unwrap();
            let var = Variable::GlobalScalar(read_id, elem);
            self.debug_var_name(read_id, ir_var);
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
        let item_id = item.id(self);
        let array_ty = self.id();
        let len_id = self.const_u32(arr.length as u32);

        self.type_array_id(Some(array_ty), item_id, len_id);
        let pointer_ty = self.type_pointer(None, StorageClass::Function, array_ty);

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
