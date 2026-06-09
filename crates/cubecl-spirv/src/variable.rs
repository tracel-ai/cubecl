#![allow(unknown_lints, unnecessary_transmutes)]

use std::mem::transmute;

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
};
use cubecl_core::ir::{self, ConstantValue, Id};
use rspirv::{
    dr::Builder,
    spirv::{self, FPEncoding, MemoryAccess, StorageClass, Word},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    GlobalBuffer(Word, Item, u32),
    GlobalScalar(Word, Elem),
    Constant(Word, ConstVal, Item),
    Local {
        id: Word,
        item: Item,
    },
    Versioned {
        id: (Id, u16),
        item: Item,
        variable: ir::Variable,
    },
    LocalBinding {
        id: Id,
        item: Item,
        variable: ir::Variable,
    },
    Raw(Word, Item),
    Slice {
        ptr: Box<Variable>,
        offset: Word,
        end: Word,
        const_len: Option<u32>,
        item: Item,
    },
    Shared(Word, Item),
    ConstantArray(Word, Item, u32),
    CoopMatrix(Id, Elem),
    Id(Word),
    Builtin(Word, Item),
}

impl Variable {
    pub fn scope(&self) -> spirv::Scope {
        if let Item::Pointer(class, _) = self.item() {
            return match class {
                StorageClass::StorageBuffer
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::Uniform => spirv::Scope::Device,
                StorageClass::Workgroup => spirv::Scope::Workgroup,
                _ => spirv::Scope::Invocation,
            };
        }
        match self {
            Variable::GlobalBuffer(..) | Variable::GlobalScalar(..) => spirv::Scope::Device,
            Variable::Shared(..) => spirv::Scope::Workgroup,
            Variable::CoopMatrix(..) => spirv::Scope::Subgroup,
            Variable::Slice { ptr, .. } => ptr.scope(),
            Variable::Raw(..) => unimplemented!("Can't get scope of raw variable"),
            Variable::Id(_) => unimplemented!("Can't get scope of raw id"),
            _ => spirv::Scope::Invocation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstVal {
    Bit32(u32),
    Bit64(u64),
}

impl ConstVal {
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstVal::Bit32(val) => *val as u64,
            ConstVal::Bit64(val) => *val,
        }
    }

    pub fn as_u32(&self) -> u32 {
        match self {
            ConstVal::Bit32(val) => *val,
            ConstVal::Bit64(_) => panic!("Truncating 64 bit variable to 32 bit"),
        }
    }

    pub fn as_float(&self, width: u32, encoding: Option<FPEncoding>) -> f64 {
        match (width, encoding) {
            (64, _) => f64::from_bits(self.as_u64()),
            (32, _) => f32::from_bits(self.as_u32()) as f64,
            (16, None) => half::f16::from_bits(self.as_u32() as u16).to_f64(),
            (_, Some(FPEncoding::BFloat16KHR)) => {
                half::bf16::from_bits(self.as_u32() as u16).to_f64()
            }
            (_, Some(FPEncoding::Float8E4M3EXT)) => {
                cubecl_common::e4m3::from_bits(self.as_u32() as u8).to_f64()
            }
            (_, Some(FPEncoding::Float8E5M2EXT)) => {
                cubecl_common::e5m2::from_bits(self.as_u32() as u8).to_f64()
            }
            _ => unreachable!(),
        }
    }

    pub fn as_int(&self, width: u32) -> i64 {
        unsafe {
            match width {
                64 => transmute::<u64, i64>(self.as_u64()),
                32 => transmute::<u32, i32>(self.as_u32()) as i64,
                16 => transmute::<u16, i16>(self.as_u32() as u16) as i64,
                8 => transmute::<u8, i8>(self.as_u32() as u8) as i64,
                _ => unreachable!(),
            }
        }
    }

    pub fn from_float(value: f64, width: u32, encoding: Option<FPEncoding>) -> Self {
        match (width, encoding) {
            (64, _) => ConstVal::Bit64(value.to_bits()),
            (32, _) => ConstVal::Bit32((value as f32).to_bits()),
            (16, None) => ConstVal::Bit32(half::f16::from_f64(value).to_bits() as u32),
            (_, Some(FPEncoding::BFloat16KHR)) => {
                ConstVal::Bit32(half::bf16::from_f64(value).to_bits() as u32)
            }
            (_, Some(FPEncoding::Float8E4M3EXT)) => {
                ConstVal::Bit32(cubecl_common::e4m3::from_f64(value).to_bits() as u32)
            }
            (_, Some(FPEncoding::Float8E5M2EXT)) => {
                ConstVal::Bit32(cubecl_common::e5m2::from_f64(value).to_bits() as u32)
            }
            _ => unreachable!(),
        }
    }

    pub fn from_int(value: i64, width: u32) -> Self {
        match width {
            64 => ConstVal::Bit64(unsafe { transmute::<i64, u64>(value) }),
            32 => ConstVal::Bit32(unsafe { transmute::<i32, u32>(value as i32) }),
            16 => ConstVal::Bit32(unsafe { transmute::<i16, u16>(value as i16) } as u32),
            8 => ConstVal::Bit32(unsafe { transmute::<i8, u8>(value as i8) } as u32),
            _ => unreachable!(),
        }
    }

    pub fn from_uint(value: u64, width: u32) -> Self {
        match width {
            64 => ConstVal::Bit64(value),
            32 => ConstVal::Bit32(value as u32),
            16 => ConstVal::Bit32(value as u16 as u32),
            8 => ConstVal::Bit32(value as u8 as u32),
            _ => unreachable!(),
        }
    }

    pub fn from_bool(value: bool) -> Self {
        ConstVal::Bit32(value as u32)
    }
}

impl From<(ConstantValue, Item)> for ConstVal {
    fn from((value, ty): (ConstantValue, Item)) -> Self {
        let elem = ty.elem();
        let width = elem.size() * 8;
        match value {
            ConstantValue::Int(val) => ConstVal::from_int(val, width),
            ConstantValue::Float(val) => ConstVal::from_float(val, width, elem.float_encoding()),
            ConstantValue::UInt(val) => ConstVal::from_uint(val, width),
            ConstantValue::Bool(val) => ConstVal::from_bool(val),
        }
    }
}

impl From<u32> for ConstVal {
    fn from(value: u32) -> Self {
        ConstVal::Bit32(value)
    }
}

impl From<f32> for ConstVal {
    fn from(value: f32) -> Self {
        ConstVal::Bit32(value.to_bits())
    }
}

impl Variable {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        match self {
            Variable::GlobalBuffer(id, _, _) => *id,
            Variable::GlobalScalar(id, _) => *id,
            Variable::Constant(id, _, _) => *id,
            Variable::Local { id, .. } => *id,
            Variable::Versioned {
                id, variable: var, ..
            } => b.get_versioned(*id, var),
            Variable::LocalBinding {
                id, variable: var, ..
            } => b.get_binding(*id, var),
            Variable::Raw(id, _) => *id,
            Variable::Slice { ptr, .. } => ptr.id(b),
            Variable::Shared(id, _) => *id,
            Variable::ConstantArray(id, _, _) => *id,
            Variable::CoopMatrix(_, _) => unimplemented!("Can't get ID from matrix var"),
            Variable::Id(id) => *id,
            Variable::Builtin(id, ..) => *id,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalBuffer(_, item, _) => item.clone(),
            Variable::GlobalScalar(_, elem) => Item::Scalar(*elem),
            Variable::Constant(_, _, item) => item.clone(),
            Variable::Local { item, .. } => item.clone(),
            Variable::Versioned { item, .. } => item.clone(),
            Variable::LocalBinding { item, .. } => item.clone(),
            Variable::Slice { item, .. } => item.clone(),
            Variable::Shared(_, item) => item.clone(),
            Variable::ConstantArray(_, item, _) => item.clone(),
            Variable::CoopMatrix(_, elem) => Item::Scalar(*elem),
            Variable::Builtin(_, item) => item.clone(),
            Variable::Raw(_, item) => item.clone(),
            Variable::Id(_) => unimplemented!("Can't get item of raw ID"),
        }
    }

    pub fn indexed_item(&self) -> Item {
        match self {
            Variable::LocalBinding {
                item: Item::Vector(elem, _),
                ..
            } => Item::Scalar(*elem),
            Variable::Local {
                item: Item::Vector(elem, _),
                ..
            } => Item::Scalar(*elem),
            Variable::Versioned {
                item: Item::Vector(elem, _),
                ..
            } => Item::Scalar(*elem),
            other => other.item(),
        }
    }

    pub fn elem(&self) -> Elem {
        self.item().elem()
    }

    pub fn has_len(&self) -> bool {
        matches!(
            self,
            Variable::GlobalBuffer(_, _, _)
                | Variable::Slice { .. }
                | Variable::ConstantArray(_, _, _)
        ) || self.item().is_array()
    }

    pub fn has_buffer_len(&self) -> bool {
        matches!(self, Variable::GlobalBuffer(_, _, _))
    }

    pub fn as_const(&self) -> Option<ConstVal> {
        match self {
            Self::Constant(_, val, _) => Some(*val),
            _ => None,
        }
    }

    pub fn as_binding(&self) -> Option<Id> {
        match self {
            Self::LocalBinding { id, .. } => Some(*id),
            _ => None,
        }
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_variable(&mut self, variable: ir::Variable) -> Variable {
        let item = variable.ty;
        match variable.kind {
            ir::VariableKind::Constant(value) => {
                let item = self.compile_type(item);
                let const_val = (value, item.clone()).into();

                if let Some(existing) = self.state.constants.get(&(const_val, item.clone())) {
                    Variable::Constant(*existing, const_val, item)
                } else {
                    let id = item.constant(self, const_val);
                    self.state.constants.insert((const_val, item.clone()), id);
                    Variable::Constant(id, const_val, item)
                }
            }
            ir::VariableKind::GlobalBuffer(pos) => {
                let buffer = self.state.buffers[pos as usize];
                Variable::GlobalBuffer(buffer.id, self.compile_type(item), pos)
            }
            ir::VariableKind::GlobalScalar(id) => self.global_scalar(id, item.storage_type()),
            ir::VariableKind::LocalMut { id } => {
                let item = self.compile_type(item);
                let var = self.get_local(id, &item, variable);
                Variable::Local { id: var, item }
            }
            ir::VariableKind::Versioned { id, version } => {
                let item = self.compile_type(item);
                let id = (id, version);
                Variable::Versioned { id, item, variable }
            }
            ir::VariableKind::LocalConst { id } => {
                let item = self.compile_type(item);
                Variable::LocalBinding { id, item, variable }
            }
            ir::VariableKind::Builtin(builtin) => {
                let item = self.compile_type(item);
                self.compile_builtin(builtin, item)
            }
            ir::VariableKind::ConstantArray { id, length, .. } => {
                let item = self.compile_type(item);
                let id = self.state.const_arrays[id as usize].id;
                Variable::ConstantArray(id, item, length as u32)
            }
            ir::VariableKind::Shared { id, .. } => {
                let item = self.compile_type(item);
                let id = self.state.shared[&id].id;
                Variable::Shared(id, item)
            }
            ir::VariableKind::Matrix { id, mat } => {
                let elem = self.compile_type(ir::Type::new(mat.storage)).elem();
                if self.state.matrices.contains_key(&id) {
                    Variable::CoopMatrix(id, elem)
                } else {
                    let matrix = self.init_coop_matrix(mat, variable, None);
                    self.state.matrices.insert(id, matrix);
                    Variable::CoopMatrix(id, elem)
                }
            }
            ir::VariableKind::Pipeline { .. } => panic!("Pipeline not supported."),
            ir::VariableKind::BarrierToken { .. } => {
                panic!("Barrier not supported.")
            }
            ir::VariableKind::TensorMap(_) => panic!("Tensor map not supported."),
            ir::VariableKind::Aggregate { .. } => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    pub fn read(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::Slice { ptr, .. } => self.read(ptr),
            Variable::Shared(id, item)
                if self.compilation_options.vulkan.supports_explicit_smem =>
            {
                let align = item.size();
                let ty = item.id(self);
                let ptr_ty =
                    Item::Pointer(StorageClass::Workgroup, Box::new(item.clone())).id(self);
                let index = self.const_u32(0);
                let access = self
                    .in_bounds_access_chain(ptr_ty, None, *id, [index])
                    .unwrap();
                self.load(
                    ty,
                    None,
                    access,
                    Some(MemoryAccess::ALIGNED),
                    [align.into()],
                )
                .unwrap()
            }
            Variable::Local { id, item } | Variable::Shared(id, item) => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, []).unwrap()
            }
            ssa => ssa.id(self),
        }
    }

    pub fn read_as(&mut self, variable: &Variable, item: &Item) -> Word {
        if let Some(as_const) = variable.as_const() {
            self.static_cast(as_const, &variable.elem(), item).0
        } else {
            let id = self.read(variable);
            variable.item().cast_to(self, None, id, item)
        }
    }

    pub fn index(&mut self, variable: &Variable, index: &Variable, out: &Variable) -> Word {
        let index_id = self.read(index);
        let write_id = self.write_id(out);
        let ptr_ty = out.item().id(self);
        let access_chain = |this, id, indices| {
            Builder::in_bounds_access_chain(this, ptr_ty, Some(write_id), id, indices).unwrap()
        };
        match variable {
            Variable::GlobalBuffer(id, ..) => {
                let zero = self.const_u32(0);
                access_chain(self, *id, vec![zero, index_id])
            }
            Variable::Slice { ptr, offset, .. } => {
                let item = Item::Scalar(Elem::Int(32, false));
                let int = item.id(self);
                let index = match index.as_const() {
                    Some(ConstVal::Bit32(0)) => *offset,
                    _ => self.i_add(int, None, *offset, index_id).unwrap(),
                };
                self.index(ptr, &Variable::Raw(index, item), out)
            }
            Variable::Shared(id, ..) if variable.item().is_array() => {
                let mut index = vec![index_id];
                if self.compilation_options.vulkan.supports_explicit_smem {
                    index.insert(0, self.const_u32(0));
                }
                access_chain(self, *id, index)
            }
            Variable::ConstantArray(id, ..) | Variable::Local { id, .. }
                if variable.item().is_array() =>
            {
                access_chain(self, *id, vec![index_id])
            }
            var => unimplemented!("Can't index into {var:?}"),
        }
    }

    pub fn write_id(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::LocalBinding { id, variable, .. } => self.get_binding(*id, variable),
            Variable::Versioned { id, variable, .. } => self.get_versioned(*id, variable),
            Variable::Local { .. } => self.id(),
            Variable::Shared(..) => self.id(),
            Variable::GlobalScalar(id, _) => *id,
            Variable::Raw(id, _) => *id,
            Variable::Constant(_, _, _) => panic!("Can't write to constant scalar"),
            Variable::GlobalBuffer(_, _, _)
            | Variable::Slice { .. }
            | Variable::ConstantArray(_, _, _) => panic!("Can't write to unindexed array"),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    pub fn write(&mut self, variable: &Variable, value: Word) {
        match variable {
            Variable::Shared(id, item)
                if self.compilation_options.vulkan.supports_explicit_smem =>
            {
                let align = item.size();
                let ptr_ty =
                    Item::Pointer(StorageClass::Workgroup, Box::new(item.clone())).id(self);
                let index = self.const_u32(0);
                let access = self
                    .in_bounds_access_chain(ptr_ty, None, *id, [index])
                    .unwrap();
                self.store(access, value, Some(MemoryAccess::ALIGNED), [align.into()])
                    .unwrap()
            }
            Variable::Local { id, .. } | Variable::Shared(id, _) => {
                self.store(*id, value, None, []).unwrap()
            }

            Variable::Slice { ptr, .. } => self.write(ptr, value),
            _ => {}
        }
    }

    pub fn load_aligned(&mut self, ptr: &Variable, out: &Variable) -> Word {
        let out_ty = out.item().id(self);
        let write_id = self.write_id(out);
        let align = ptr.item().size();

        let ptr = self.read(ptr);

        self.load(
            out_ty,
            Some(write_id),
            ptr,
            Some(MemoryAccess::ALIGNED),
            [align.into()],
        )
        .unwrap()
    }

    pub fn store_aligned(&mut self, ptr: &Variable, value: &Variable) {
        let align = ptr.item().size();

        let ptr = self.read(ptr);
        let value = self.read(value);

        self.store(ptr, value, Some(MemoryAccess::ALIGNED), [align.into()])
            .unwrap()
    }
}
