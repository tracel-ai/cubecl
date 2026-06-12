use core::ops::{Deref, DerefMut};
use std::collections::VecDeque;

use cubecl_core::{
    ir::{self, Builtin, Id, ValueKind},
    prelude::KernelDefinition,
};
use cubecl_opt::NodeIndex;
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
    value::ConstVal,
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
    pub shared: HashMap<Id, SharedVal>,

    pub globals: HashMap<Builtin, Word>,
    pub loaded_builtins: HashMap<BuiltIn, Word>,
    pub used_builtins: HashMap<BuiltIn, (Word, Item)>,

    pub constants: HashMap<(ConstVal, Item), Word>,
    pub values: HashMap<Id, Word>,
    pub labels: HashMap<NodeIndex, Word>,
    pub end_labels: HashMap<NodeIndex, Word>,
}

#[derive(Clone, Debug)]
pub struct FuncDefinition {
    pub type_id: Word,
    pub id: Word,
}

#[derive(Clone, Debug)]
pub struct SharedVal {
    pub id: Word,
    pub val_id: Word,
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
    pub arr_ty_id: Word,
    pub arr_ptr_ty_id: Word,
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
            if binding.value.ty.vector_size() > max_vector_size {
                binding.value.ty = binding.value.ty.with_vector_size(max_vector_size);
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

            let smem_val_id = if self.compilation_options.vulkan.supports_explicit_smem {
                self.id()
            } else {
                smem_id
            };

            let item = self.compile_type(alloc.smem.value_ty);
            self.state.base_lookups.shared.insert(
                alloc.id,
                SharedVal {
                    id: smem_id,
                    val_id: smem_val_id,
                    ptr_ty_id: smem_ptr_ty_id,
                    item,
                    offset: alloc.offset as u32,
                    align: alloc.smem.alignment as u32,
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
        self.state.lookups = self.state.base_lookups.clone();
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

    pub fn get_value(&mut self, id: Id) -> Word {
        if let Some(existing) = self.state.values.get(&id) {
            *existing
        } else {
            let word = self.id();
            self.state.values.insert(id, word);
            self.debug_val_name(word, id);
            word
        }
    }

    pub fn insert_value(&mut self, id: Id, word: Word) {
        self.state.values.insert(id, word);
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

    pub fn init_function_param(&mut self, param: ir::ExpandValue, param_id: Word) {
        let item = self.compile_type(param.ty);
        match param.kind {
            ValueKind::Constant(value) => {
                let const_val = (value, item.clone()).into();
                self.state.constants.insert((const_val, item), param_id);
            }
            ValueKind::Value { id } => {
                self.state.values.insert(id, param_id);
            }
        }
    }

    pub fn end_function_and_reset_lookups(&mut self) {
        self.builder.end_function().unwrap();
        self.state.lookups = self.state.base_lookups.clone();
    }

    pub fn global_scalar(&mut self, id: Id, ty: ir::StorageType) -> Word {
        self.insert_in_setup(|b| {
            let field_id = b.const_u32(b.state.scalar_bindings[&ty]);
            let offset = b.const_u32(id);
            let item = b.compile_type(ir::Type::new(ty));
            let align = item.size();
            let ty_id = item.id(b);
            let storage_class = T::info_storage_class(b);
            let ptr_ty = Item::Pointer(storage_class, Box::new(item)).id(b);
            let info = b.state.info.unwrap().id;
            let access = b
                .in_bounds_access_chain(ptr_ty, None, info, [field_id, offset])
                .unwrap();
            b.load(
                ty_id,
                None,
                access,
                Some(MemoryAccess::ALIGNED),
                [align.into()],
            )
            .unwrap()
        })
    }
}
