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

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Constant(Word, ConstVal, Item),
    Value { id: Id, item: Item },
    Raw(Word, Item),
    Id(Word),
}

impl Value {
    pub fn scope(&self) -> spirv::Scope {
        match self.item() {
            Item::Pointer(class, _) => {
                return match class {
                    StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::Uniform => spirv::Scope::Device,
                    StorageClass::Workgroup => spirv::Scope::Workgroup,
                    _ => spirv::Scope::Invocation,
                };
            }
            Item::CoopMatrix { scope, .. } => {
                return scope;
            }
            _ => {}
        }
        match self {
            Value::Raw(..) => unimplemented!("Can't get scope of raw variable"),
            Value::Id(_) => unimplemented!("Can't get scope of raw id"),
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

impl Value {
    pub fn id<T: SpirvTarget>(&self, b: &mut SpirvCompiler<T>) -> Word {
        match self {
            Value::Constant(id, _, _) => *id,
            Value::Value { id, .. } => b.get_value(*id),
            Value::Raw(id, _) => *id,
            Value::Id(id) => *id,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Value::Constant(_, _, item) => item.clone(),
            Value::Value { item, .. } => item.clone(),
            Value::Raw(_, item) => item.clone(),
            Value::Id(_) => unimplemented!("Can't get item of raw ID"),
        }
    }

    pub fn indexed_item(&self) -> Item {
        match self {
            Value::Value {
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
        self.item().is_array()
    }

    pub fn has_buffer_len(&self) -> bool {
        matches!(self.item(), Item::DynamicArray(..))
    }

    pub fn as_const(&self) -> Option<ConstVal> {
        match self {
            Self::Constant(_, val, _) => Some(*val),
            _ => None,
        }
    }

    pub fn as_binding(&self) -> Option<Id> {
        match self {
            Self::Value { id, .. } => Some(*id),
            _ => None,
        }
    }
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_value(&mut self, value: ir::Value) -> Value {
        let item = value.ty;
        match value.kind {
            ir::ValueKind::Constant(value) => {
                let item = self.compile_type(item);
                let const_val = (value, item.clone()).into();

                if let Some(existing) = self.state.constants.get(&(const_val, item.clone())) {
                    Value::Constant(*existing, const_val, item)
                } else {
                    let id = item.constant(self, const_val);
                    self.state.constants.insert((const_val, item.clone()), id);
                    Value::Constant(id, const_val, item)
                }
            }
            ir::ValueKind::Value { id } => {
                let item = self.compile_type(item);
                Value::Value { id, item }
            }
        }
    }

    pub fn read(&mut self, variable: &Value) -> Word {
        variable.id(self)
    }

    pub fn read_as(&mut self, variable: &Value, item: &Item) -> Word {
        if let Some(as_const) = variable.as_const() {
            self.static_cast(as_const, &variable.elem(), item).0
        } else {
            let id = self.read(variable);
            variable.item().cast_to(self, None, id, item)
        }
    }

    pub fn index(&mut self, variable: &Value, index: &Value, out: &Value) -> Word {
        let index_id = self.read(index);
        let write_id = self.write_id(out);
        let ptr_ty = out.item().id(self);
        let access_chain = |this, id, indices| {
            Builder::in_bounds_access_chain(this, ptr_ty, Some(write_id), id, indices).unwrap()
        };
        let list = self.read(variable);
        access_chain(self, list, vec![index_id])
    }

    pub fn write_id(&mut self, value: &Value) -> Word {
        match value {
            Value::Value { id, .. } => self.get_value(*id),
            Value::Raw(id, _) => *id,
            Value::Constant(_, _, _) => panic!("Can't write to constant scalar"),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    /// Like [`write_id`], but specialized for the hacky semantics of CMMA ops. Unlike normal ops, they
    /// can have mutable variables as direct outputs.
    pub fn write_id_cmma(&mut self, value: &Value) -> Word {
        match value {
            Value::Value { item, .. } if item.is_ptr() => self.id(),
            Value::Value { id, .. } => self.get_value(*id),
            Value::Raw(id, _) => *id,
            Value::Constant(_, _, _) => panic!("Can't write to constant scalar"),
            global => panic!("Can't write to builtin {global:?}"),
        }
    }

    pub fn write(&mut self, variable: &Value, value: Word) {
        if let Value::Value { id, .. } = variable {
            self.state.values.insert(*id, value);
        }
    }

    /// Like [`write`], but specialized for the hacky semantics of CMMA ops. Unlike normal ops, they
    /// can have mutable variables as direct outputs.
    pub fn write_cmma(&mut self, variable: &Value, value: Word) {
        match variable {
            ptr @ Value::Value { item, .. } if item.is_ptr() => {
                let ptr = self.read(ptr);

                self.store(ptr, value, None, []).unwrap()
            }
            Value::Value { id, .. } => {
                self.state.values.insert(*id, value);
            }
            _ => {}
        }
    }

    pub fn load_aligned(&mut self, ptr: &Value, out: &Value) -> Word {
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

    pub fn store_aligned(&mut self, ptr: &Value, value: &Value) {
        let align = ptr.item().size();

        let ptr = self.read(ptr);
        let value = self.read(value);

        self.store(ptr, value, Some(MemoryAccess::ALIGNED), [align.into()])
            .unwrap()
    }
}
