use std::mem::transmute;

use crate::{
    item::{Elem, Item},
    lookups::Array,
    SpirvCompiler, SpirvTarget,
};
use cubecl_core::{
    ir::{self as core, ConstantScalarValue, FloatKind},
    ExecutionMode,
};
use rspirv::{
    dr::Builder,
    spirv::{StorageClass, Word},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    SubgroupSize(Word),
    GlobalInputArray(Word, Item),
    GlobalOutputArray(Word, Item),
    GlobalScalar(Word, Elem),
    ConstantScalar(Word, ConstVal, Elem),
    Local {
        id: Word,
        item: Item,
    },
    Versioned {
        id: (u16, u8, u16),
        item: Item,
    },
    LocalBinding {
        id: (u16, u8),
        item: Item,
    },
    Raw(Word, Item),
    Named {
        id: Word,
        item: Item,
        is_array: bool,
    },
    Slice {
        ptr: Box<Variable>,
        offset: Word,
        end: Word,
        const_len: Option<u32>,
        item: Item,
    },
    SharedMemory(Word, Item, u32),
    ConstantArray(Word, Item, u32),
    LocalArray(Word, Item, u32),
    CoopMatrix(u16, u8, Elem),
    Id(Word),
    LocalInvocationIndex(Word),
    LocalInvocationIdX(Word),
    LocalInvocationIdY(Word),
    LocalInvocationIdZ(Word),
    Rank(Word),
    WorkgroupId(Word),
    WorkgroupIdX(Word),
    WorkgroupIdY(Word),
    WorkgroupIdZ(Word),
    GlobalInvocationIndex(Word),
    GlobalInvocationIdX(Word),
    GlobalInvocationIdY(Word),
    GlobalInvocationIdZ(Word),
    WorkgroupSize(Word),
    WorkgroupSizeX(Word),
    WorkgroupSizeY(Word),
    WorkgroupSizeZ(Word),
    NumWorkgroups(Word),
    NumWorkgroupsX(Word),
    NumWorkgroupsY(Word),
    NumWorkgroupsZ(Word),
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

    pub fn as_float(&self, width: u32) -> f64 {
        match width {
            64 => f64::from_bits(self.as_u64()),
            32 => f32::from_bits(self.as_u32()) as f64,
            16 => half::f16::from_bits(self.as_u32() as u16).to_f64(),
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

    pub fn from_float(value: f64, width: u32) -> Self {
        match width {
            64 => ConstVal::Bit64(value.to_bits()),
            32 => ConstVal::Bit32((value as f32).to_bits()),
            16 => ConstVal::Bit32(half::f16::from_f64(value).to_bits() as u32),
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

impl From<ConstantScalarValue> for ConstVal {
    fn from(value: ConstantScalarValue) -> Self {
        let width = value.elem().size() as u32 * 8;
        match value {
            ConstantScalarValue::Int(val, _) => ConstVal::from_int(val, width),
            ConstantScalarValue::Float(_, FloatKind::BF16) => {
                panic!("bf16 not supported in SPIR-V")
            }
            ConstantScalarValue::Float(val, _) => ConstVal::from_float(val, width),
            ConstantScalarValue::UInt(val, _) => ConstVal::from_uint(val, width),
            ConstantScalarValue::Bool(val) => ConstVal::from_bool(val),
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
            Variable::GlobalInputArray(id, _) => *id,
            Variable::GlobalOutputArray(id, _) => *id,
            Variable::GlobalScalar(id, _) => *id,
            Variable::ConstantScalar(id, _, _) => *id,
            Variable::Local { id, .. } => *id,
            Variable::Versioned { id, .. } => b.get_versioned(*id),
            Variable::LocalBinding { id, .. } => b.get_binding(*id),
            Variable::Raw(id, _) => *id,
            Variable::Named { id, .. } => *id,
            Variable::Slice { ptr, .. } => ptr.id(b),
            Variable::SharedMemory(id, _, _) => *id,
            Variable::ConstantArray(id, _, _) => *id,
            Variable::LocalArray(id, _, _) => *id,
            Variable::CoopMatrix(_, _, _) => unimplemented!("Can't get ID from matrix var"),
            Variable::SubgroupSize(id) => *id,
            Variable::Id(id) => *id,
            Variable::LocalInvocationIndex(id) => *id,
            Variable::LocalInvocationIdX(id) => *id,
            Variable::LocalInvocationIdY(id) => *id,
            Variable::LocalInvocationIdZ(id) => *id,
            Variable::Rank(id) => *id,
            Variable::WorkgroupId(id) => *id,
            Variable::WorkgroupIdX(id) => *id,
            Variable::WorkgroupIdY(id) => *id,
            Variable::WorkgroupIdZ(id) => *id,
            Variable::GlobalInvocationIndex(id) => *id,
            Variable::GlobalInvocationIdX(id) => *id,
            Variable::GlobalInvocationIdY(id) => *id,
            Variable::GlobalInvocationIdZ(id) => *id,
            Variable::WorkgroupSize(id) => *id,
            Variable::WorkgroupSizeX(id) => *id,
            Variable::WorkgroupSizeY(id) => *id,
            Variable::WorkgroupSizeZ(id) => *id,
            Variable::NumWorkgroups(id) => *id,
            Variable::NumWorkgroupsX(id) => *id,
            Variable::NumWorkgroupsY(id) => *id,
            Variable::NumWorkgroupsZ(id) => *id,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray(_, item) => item.clone(),
            Variable::GlobalOutputArray(_, item) => item.clone(),
            Variable::GlobalScalar(_, elem) => Item::Scalar(*elem),
            Variable::ConstantScalar(_, _, elem) => Item::Scalar(*elem),
            Variable::Local { item, .. } => item.clone(),
            Variable::Versioned { item, .. } => item.clone(),
            Variable::LocalBinding { item, .. } => item.clone(),
            Variable::Named { item, .. } => item.clone(),
            Variable::Slice { item, .. } => item.clone(),
            Variable::SharedMemory(_, item, _) => item.clone(),
            Variable::ConstantArray(_, item, _) => item.clone(),
            Variable::LocalArray(_, item, _) => item.clone(),
            Variable::CoopMatrix(_, _, elem) => Item::Scalar(*elem),
            _ => Item::Scalar(Elem::Int(32, false)), // builtin
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
            Variable::GlobalInputArray(_, _)
                | Variable::GlobalOutputArray(_, _)
                | Variable::Named {
                    is_array: false,
                    ..
                }
                | Variable::Slice { .. }
                | Variable::SharedMemory(_, _, _)
                | Variable::ConstantArray(_, _, _)
                | Variable::LocalArray(_, _, _)
        )
    }

    pub fn as_const(&self) -> Option<ConstVal> {
        match self {
            Self::ConstantScalar(_, val, _) => Some(*val),
            _ => None,
        }
    }

    pub fn as_binding(&self) -> Option<(u16, u8)> {
        match self {
            Self::LocalBinding { id, .. } => Some(*id),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum IndexedVariable {
    Pointer(Word, Item),
    Composite(Word, u32, Item),
    Scalar(Variable),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Globals {
    Rank2,

    Id,
    LocalInvocationId,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    LocalInvocationIndex,
    Rank,
    WorkgroupId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    WorkgroupIndex,
    GlobalInvocationId,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    GlobalInvocationIndex,
    WorkgroupSize,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroups,
    NumWorkgroupsTotal,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    SubgroupSize,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_variable(&mut self, variable: core::Variable) -> Variable {
        let item = variable.item;
        match variable.kind {
            core::VariableKind::ConstantScalar(value) => {
                let item = self.compile_item(core::Item::new(value.elem()));
                let const_val = value.into();

                if let Some(existing) = self.state.constants.get(&(const_val, item.clone())) {
                    Variable::ConstantScalar(*existing, const_val, item.elem())
                } else {
                    let id = item.elem().constant(self, const_val);
                    self.state.constants.insert((const_val, item.clone()), id);
                    Variable::ConstantScalar(id, const_val, item.elem())
                }
            }
            core::VariableKind::Slice { id, depth, .. } => self
                .state
                .slices
                .get(&(id, depth))
                .expect("Tried accessing non-existing slice")
                .into(),
            core::VariableKind::GlobalInputArray(id) => {
                let id = self.state.inputs[id as usize];
                Variable::GlobalInputArray(id, self.compile_item(item))
            }
            core::VariableKind::GlobalOutputArray(id) => {
                let id = self.state.outputs[id as usize];
                Variable::GlobalOutputArray(id, self.compile_item(item))
            }
            core::VariableKind::GlobalScalar(id) => self.global_scalar(id, item.elem),
            core::VariableKind::Local { id, depth } => {
                let item = self.compile_item(item);
                let var = self.get_local((id, depth), &item);
                Variable::Local { id: var, item }
            }
            core::VariableKind::Versioned { id, depth, version } => {
                let item = self.compile_item(item);
                let id = (id, depth, version);
                Variable::Versioned { id, item }
            }
            core::VariableKind::LocalBinding { id, depth } => {
                let item = self.compile_item(item);
                let id = (id, depth);
                Variable::LocalBinding { id, item }
            }
            core::VariableKind::Builtin(builtin) => self.compile_builtin(builtin),
            core::VariableKind::ConstantArray { id, length } => {
                let item = self.compile_item(item);
                let id = self.state.const_arrays[id as usize].id;
                Variable::ConstantArray(id, item, length)
            }
            core::VariableKind::SharedMemory { id, length } => {
                let item = self.compile_item(item);
                let id = if let Some(arr) = self.state.shared_memories.get(&id) {
                    arr.id
                } else {
                    let arr_id = self.id();
                    let arr = Array {
                        id: arr_id,
                        item: item.clone(),
                        len: length,
                    };
                    self.state.shared_memories.insert(id, arr);
                    arr_id
                };
                Variable::SharedMemory(id, item, length)
            }
            core::VariableKind::LocalArray { id, depth, length } => {
                let item = self.compile_item(item);
                let id = if let Some(arr) = self.state.local_arrays.get(&(id, depth)) {
                    arr.id
                } else {
                    let arr_ty = Item::Array(Box::new(item.clone()), length);
                    let ptr_ty = Item::Pointer(StorageClass::Function, Box::new(arr_ty)).id(self);
                    let arr_id = self.declare_function_variable(ptr_ty);
                    self.debug_name(arr_id, format!("array({id}, {depth})"));
                    let arr = Array {
                        id: arr_id,
                        item: item.clone(),
                        len: length,
                    };
                    self.state.local_arrays.insert((id, depth), arr);
                    arr_id
                };
                Variable::LocalArray(id, item, length)
            }
            core::VariableKind::Matrix { id, mat, depth } => {
                let elem = self.compile_item(core::Item::new(mat.elem)).elem();
                if self.state.matrices.contains_key(&(id, depth)) {
                    Variable::CoopMatrix(id, depth, elem)
                } else {
                    let matrix = self.init_coop_matrix(mat);
                    self.state.matrices.insert((id, depth), matrix);
                    Variable::CoopMatrix(id, depth, elem)
                }
            }
        }
    }

    pub fn rank(&mut self) -> Word {
        self.get_or_insert_global(Globals::Rank, |b| {
            let int = Item::Scalar(Elem::Int(32, false));
            let int_ty = int.id(b);
            let int_ptr = Item::Pointer(StorageClass::StorageBuffer, Box::new(int)).id(b);
            let info = b.state.named["info"];
            let zero = b.const_u32(0);
            let rank_ptr = b
                .access_chain(int_ptr, None, info, vec![zero, zero])
                .unwrap();
            let rank = b.load(int_ty, None, rank_ptr, None, vec![]).unwrap();
            b.debug_name(rank, "rank");
            rank
        })
    }

    pub fn rank_2(&mut self) -> Word {
        self.get_or_insert_global(Globals::Rank2, |b| {
            let int = Item::Scalar(Elem::Int(32, false));
            let int_ty = int.id(b);
            let rank = b.rank();
            let two = b.const_u32(2);
            let rank_2 = b.i_mul(int_ty, None, rank, two).unwrap();
            b.debug_name(rank_2, "rank*2");
            rank_2
        })
    }

    pub fn read(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::GlobalInputArray(id, _) => *id,
            Variable::GlobalOutputArray(id, _) => *id,
            Variable::SharedMemory(id, _, _) => *id,
            Variable::ConstantArray(id, _, _) => *id,
            Variable::LocalArray(id, _, _) => *id,
            Variable::Slice { ptr, .. } => self.read(ptr),
            Variable::Local { id, item } => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, vec![]).unwrap()
            }
            Variable::Named { id, item, .. } => {
                let ty = item.id(self);
                self.load(ty, None, *id, None, vec![]).unwrap()
            }
            ssa => ssa.id(self),
        }
    }

    pub fn read_as(&mut self, variable: &Variable, item: &Item) -> Word {
        if let Some(as_const) = variable.as_const() {
            self.static_cast(as_const, &variable.elem(), item)
        } else {
            let id = self.read(variable);
            variable.item().cast_to(self, None, id, item)
        }
    }

    pub fn index(
        &mut self,
        variable: &Variable,
        index: &Variable,
        unchecked: bool,
    ) -> IndexedVariable {
        let access_chain = if unchecked {
            Builder::in_bounds_access_chain
        } else {
            Builder::access_chain
        };
        let index_id = self.read(index);
        match variable {
            Variable::GlobalInputArray(id, item)
            | Variable::GlobalOutputArray(id, item)
            | Variable::Named { id, item, .. } => {
                let ptr_ty =
                    Item::Pointer(StorageClass::StorageBuffer, Box::new(item.clone())).id(self);
                let zero = self.const_u32(0);
                let id = access_chain(self, ptr_ty, None, *id, vec![zero, index_id]).unwrap();

                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::Local {
                id,
                item: Item::Vector(elem, _),
            } => {
                let ptr_ty =
                    Item::Pointer(StorageClass::Function, Box::new(Item::Scalar(*elem))).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();

                IndexedVariable::Pointer(id, Item::Scalar(*elem))
            }
            Variable::LocalBinding {
                id,
                item: Item::Vector(elem, vec),
            } => IndexedVariable::Composite(
                self.get_binding(*id),
                index
                    .as_const()
                    .expect("Index into vector must be constant")
                    .as_u32(),
                Item::Vector(*elem, *vec),
            ),
            Variable::Versioned {
                id,
                item: Item::Vector(elem, vec),
            } => IndexedVariable::Composite(
                self.get_versioned(*id),
                index
                    .as_const()
                    .expect("Index into vector must be constant")
                    .as_u32(),
                Item::Vector(*elem, *vec),
            ),
            Variable::LocalBinding { .. } | Variable::Local { .. } => {
                let index = index
                    .as_const()
                    .expect("Index into vector must be constant")
                    .as_u32();
                if index > 0 {
                    panic!("Tried accessing {index}th element of scalar!");
                } else {
                    IndexedVariable::Scalar(variable.clone())
                }
            }
            Variable::Slice { ptr, offset, .. } => {
                let item = Item::Scalar(Elem::Int(32, false));
                let int = item.id(self);
                let index = match index.as_const() {
                    Some(ConstVal::Bit32(0)) => *offset,
                    _ => self.i_add(int, None, *offset, index_id).unwrap(),
                };
                self.index(ptr, &Variable::Raw(index, item), unchecked)
            }
            Variable::SharedMemory(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::Workgroup, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::ConstantArray(id, item, _) => {
                let ptr_ty =
                    Item::Pointer(StorageClass::UniformConstant, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            Variable::LocalArray(id, item, _) => {
                let ptr_ty = Item::Pointer(StorageClass::Function, Box::new(item.clone())).id(self);
                let id = access_chain(self, ptr_ty, None, *id, vec![index_id]).unwrap();
                IndexedVariable::Pointer(id, item.clone())
            }
            var => unimplemented!("Can't index into {var:?}"),
        }
    }

    pub fn read_indexed(&mut self, out: &Variable, variable: &Variable, index: &Variable) -> Word {
        let checked = matches!(self.mode, ExecutionMode::Checked) && variable.has_len();
        let always_in_bounds = is_always_in_bounds(variable, index);
        let index_id = self.read(index);
        let indexed = self.index(variable, index, always_in_bounds);

        let read = |b: &mut Self| match indexed {
            IndexedVariable::Pointer(ptr, item) => {
                let ty = item.id(b);
                let out_id = b.write_id(out);
                b.load(ty, Some(out_id), ptr, None, vec![]).unwrap()
            }
            IndexedVariable::Composite(var, index, item) => {
                let elem = item.elem();
                let ty = elem.id(b);
                let out_id = b.write_id(out);
                b.composite_extract(ty, Some(out_id), var, vec![index])
                    .unwrap()
            }
            IndexedVariable::Scalar(var) => {
                let ty = out.item().id(b);
                let input = b.read(&var);
                let out_id = b.write_id(out);
                b.copy_object(ty, Some(out_id), input).unwrap();
                b.write(out, out_id);
                out_id
            }
        };
        if checked && !always_in_bounds {
            self.compile_read_bound(variable, index_id, variable.item(), read)
        } else {
            read(self)
        }
    }

    pub fn read_indexed_unchecked(
        &mut self,
        out: &Variable,
        variable: &Variable,
        index: &Variable,
    ) -> Word {
        let indexed = self.index(variable, index, true);

        match indexed {
            IndexedVariable::Pointer(ptr, item) => {
                let ty = item.id(self);
                let out_id = self.write_id(out);
                self.load(ty, Some(out_id), ptr, None, vec![]).unwrap()
            }
            IndexedVariable::Composite(var, index, item) => {
                let elem = item.elem();
                let ty = elem.id(self);
                let out_id = self.write_id(out);
                self.composite_extract(ty, Some(out_id), var, vec![index])
                    .unwrap()
            }
            IndexedVariable::Scalar(var) => {
                let ty = out.item().id(self);
                let input = self.read(&var);
                let out_id = self.write_id(out);
                self.copy_object(ty, Some(out_id), input).unwrap();
                self.write(out, out_id);
                out_id
            }
        }
    }

    pub fn index_ptr(&mut self, var: &Variable, index: &Variable) -> Word {
        match self.index(var, index, false) {
            IndexedVariable::Pointer(ptr, _) => ptr,
            other => unreachable!("{other:?}"),
        }
    }

    pub fn write_id(&mut self, variable: &Variable) -> Word {
        match variable {
            Variable::LocalBinding { id, .. } => self.get_binding(*id),
            Variable::Versioned { id, .. } => self.get_versioned(*id),
            Variable::Local { .. } => self.id(),
            Variable::GlobalScalar(id, _) => *id,
            Variable::Raw(id, _) => *id,
            Variable::ConstantScalar(_, _, _) => panic!("Can't write to constant scalar"),
            Variable::GlobalInputArray(_, _)
            | Variable::GlobalOutputArray(_, _)
            | Variable::Slice { .. }
            | Variable::Named { .. }
            | Variable::SharedMemory(_, _, _)
            | Variable::ConstantArray(_, _, _)
            | Variable::LocalArray(_, _, _) => panic!("Can't write to unindexed array"),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    pub fn write(&mut self, variable: &Variable, value: Word) {
        match variable {
            Variable::Local { id, .. } => self.store(*id, value, None, vec![]).unwrap(),
            Variable::Slice { ptr, .. } => self.write(ptr, value),
            _ => {}
        }
    }

    pub fn write_indexed(&mut self, out: &Variable, index: &Variable, value: Word) {
        let checked = matches!(self.mode, ExecutionMode::Checked) && out.has_len();
        let always_in_bounds = is_always_in_bounds(out, index);
        let index_id = self.read(index);
        let variable = self.index(out, index, always_in_bounds);

        let write = |b: &mut Self| match variable {
            IndexedVariable::Pointer(ptr, _) => b.store(ptr, value, None, vec![]).unwrap(),
            IndexedVariable::Composite(var, index, item) => {
                let ty = item.id(b);
                let id = b
                    .composite_insert(ty, None, value, var, vec![index])
                    .unwrap();
                b.write(out, id);
            }
            IndexedVariable::Scalar(var) => b.write(&var, value),
        };
        if checked && !always_in_bounds {
            self.compile_write_bound(out, index_id, write);
        } else {
            write(self)
        }
    }

    pub fn write_indexed_unchecked(&mut self, out: &Variable, index: &Variable, value: Word) {
        let variable = self.index(out, index, true);

        match variable {
            IndexedVariable::Pointer(ptr, _) => self.store(ptr, value, None, vec![]).unwrap(),
            IndexedVariable::Composite(var, index, item) => {
                let ty = item.id(self);
                let out_id = self
                    .composite_insert(ty, None, value, var, vec![index])
                    .unwrap();
                self.write(out, out_id);
            }
            IndexedVariable::Scalar(var) => self.write(&var, value),
        }
    }
}

fn is_always_in_bounds(var: &Variable, index: &Variable) -> bool {
    let len = match var {
        Variable::SharedMemory(_, _, len)
        | Variable::ConstantArray(_, _, len)
        | Variable::LocalArray(_, _, len)
        | Variable::Slice {
            const_len: Some(len),
            ..
        } => *len,
        _ => return false,
    };

    let const_index = match index {
        Variable::ConstantScalar(_, value, _) => value.as_u32(),
        _ => return false,
    };

    const_index < len
}
