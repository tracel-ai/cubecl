use core::{fmt::Display, hash::Hash};

use crate::{
    FloatKind, IntKind, Scope, StorageType, TypeHash,
    attributes::{BoolAttr, FloatAttr, IntAttr, UIntAttr},
    dialect::memory::LoadOp,
    interfaces::TypedExt,
};

use super::{ElemType, Type, UIntKind};
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
use derive_more::From;
use float_ord::FloatOrd;
use pliron::{
    builtin::{op_interfaces::OneResultInterface, ops::ConstantOp},
    derive::format,
    r#type::TypePtr,
    utils::apfloat::f64_to_double,
    value::Value,
};

impl ExpandValue {
    pub fn new(value: Value) -> Self {
        Self::Value(value)
    }

    pub fn constant(value: ConstantValue, ty: impl Into<StorageType>) -> Self {
        let ty = ty.into();
        let value = value.cast_to(ty);
        Self::Constant { value, ty }
    }

    pub fn read_value(&self, scope: &Scope) -> Value {
        let val = self.value(scope);
        if val.is_ptr(scope.ctx()) {
            let op = LoadOp::new(scope.ctx_mut(), val);
            scope.register(&op);
            op.get_result(scope.ctx())
        } else {
            val
        }
    }

    pub fn value(&self, scope: &Scope) -> Value {
        match self {
            ExpandValue::Value(value) => *value,
            ExpandValue::Constant { value, ty } => {
                let ctx = scope.ctx_mut();
                let ty = ty.to_type(ctx);
                let value = match value {
                    ConstantValue::Int(value) => {
                        IntAttr::new(TypePtr::from_ptr(ty, ctx).unwrap(), *value).into()
                    }
                    ConstantValue::UInt(value) => {
                        UIntAttr::new(TypePtr::from_ptr(ty, ctx).unwrap(), *value).into()
                    }
                    ConstantValue::Float(value) => {
                        FloatAttr::new(TypePtr::from_ptr(ty, ctx).unwrap(), f64_to_double(*value))
                            .into()
                    }
                    ConstantValue::Bool(value) => BoolAttr::new(*value).into(),
                };
                let op = ConstantOp::new(scope.ctx_mut(), value);
                scope.register(&op);
                op.get_result(ctx)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
pub enum ExpandValue {
    Value(Value),
    Constant {
        value: ConstantValue,
        ty: StorageType,
    },
}

impl From<Value> for ExpandValue {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord)]
#[format]
#[repr(u32)]
pub enum Builtin {
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePosCluster,
    CubePosClusterX,
    CubePosClusterY,
    CubePosClusterZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeClusterDim,
    CubeClusterDimX,
    CubeClusterDimY,
    CubeClusterDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    PlaneDim,
    PlanePos,
    UnitPosPlane,
    AbsolutePos,
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
}

/// The scalars are stored with the highest precision possible, but they might get reduced during
/// compilation. For constant propagation, casts are always executed before converting back to the
/// larger type to ensure deterministic output.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, PartialOrd, From)]
#[allow(missing_docs, clippy::derive_ord_xor_partial_ord)]
pub enum ConstantValue {
    Int(i64),
    Float(f64),
    UInt(u64),
    Bool(bool),
}

impl Ord for ConstantValue {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Override float-float comparison with `FloatOrd` since `f64` isn't `Ord`. All other
        // comparisons are safe to unwrap since they're either `Ord` or only compare discriminants.
        match (self, other) {
            (ConstantValue::Float(this), ConstantValue::Float(other)) => {
                FloatOrd(*this).cmp(&FloatOrd(*other))
            }
            _ => self.partial_cmp(other).unwrap(),
        }
    }
}

impl Eq for ConstantValue {}
impl Hash for ConstantValue {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        core::mem::discriminant(self).hash(ra_expand_state);
        match self {
            ConstantValue::Int(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Float(f0) => {
                FloatOrd(*f0).hash(ra_expand_state);
            }
            ConstantValue::UInt(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Bool(f0) => {
                f0.hash(ra_expand_state);
            }
        }
    }
}

impl ConstantValue {
    /// Returns the value of the constant as a usize.
    ///
    /// It will return [None] if the constant type is a float or a bool.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            ConstantValue::UInt(val) => Some(*val as usize),
            ConstantValue::Int(val) => Some(*val as usize),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the constant as a usize.
    pub fn as_usize(&self) -> usize {
        match self {
            ConstantValue::UInt(val) => *val as usize,
            ConstantValue::Int(val) => *val as usize,
            ConstantValue::Float(val) => *val as usize,
            ConstantValue::Bool(val) => *val as usize,
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u32(&self) -> Option<u32> {
        self.try_as_u64().map(|it| it as u32)
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }

    /// Returns the value of the scalar as a u64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u64(&self) -> Option<u64> {
        match self {
            ConstantValue::UInt(val) => Some(*val),
            ConstantValue::Int(val) => Some(*val as u64),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u64.
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstantValue::UInt(val) => *val,
            ConstantValue::Int(val) => *val as u64,
            ConstantValue::Float(val) => *val as u64,
            ConstantValue::Bool(val) => *val as u64,
        }
    }

    /// Returns the value of the scalar as a i64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_i64(&self) -> Option<i64> {
        match self {
            ConstantValue::UInt(val) => Some(*val as i64),
            ConstantValue::Int(val) => Some(*val),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a i128.
    pub fn as_i128(&self) -> i128 {
        match self {
            ConstantValue::UInt(val) => *val as i128,
            ConstantValue::Int(val) => *val as i128,
            ConstantValue::Float(val) => *val as i128,
            ConstantValue::Bool(val) => *val as i128,
        }
    }

    /// Returns the value of the scalar as a i64.
    pub fn as_i64(&self) -> i64 {
        match self {
            ConstantValue::UInt(val) => *val as i64,
            ConstantValue::Int(val) => *val,
            ConstantValue::Float(val) => *val as i64,
            ConstantValue::Bool(val) => *val as i64,
        }
    }

    /// Returns the value of the scalar as a i64.
    pub fn as_i32(&self) -> i32 {
        match self {
            ConstantValue::UInt(val) => *val as i32,
            ConstantValue::Int(val) => *val as i32,
            ConstantValue::Float(val) => *val as i32,
            ConstantValue::Bool(val) => *val as i32,
        }
    }

    /// Returns the value of the scalar as a f64.
    ///
    /// It will return [None] if the scalar type is an int or a bool.
    pub fn try_as_f64(&self) -> Option<f64> {
        match self {
            ConstantValue::Float(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the value of the scalar as a f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            ConstantValue::UInt(val) => *val as f64,
            ConstantValue::Int(val) => *val as f64,
            ConstantValue::Float(val) => *val,
            ConstantValue::Bool(val) => *val as u8 as f64,
        }
    }

    /// Returns the value of the variable as a bool if it actually is a bool.
    pub fn try_as_bool(&self) -> Option<bool> {
        match self {
            ConstantValue::Bool(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the value of the variable as a bool.
    ///
    /// It will panic if the scalar isn't a bool.
    pub fn as_bool(&self) -> bool {
        match self {
            ConstantValue::UInt(val) => *val != 0,
            ConstantValue::Int(val) => *val != 0,
            ConstantValue::Float(val) => *val != 0.,
            ConstantValue::Bool(val) => *val,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 0,
            ConstantValue::Float(val) => *val == 0.0,
            ConstantValue::UInt(val) => *val == 0,
            ConstantValue::Bool(val) => !*val,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 1,
            ConstantValue::Float(val) => *val == 1.0,
            ConstantValue::UInt(val) => *val == 1,
            ConstantValue::Bool(val) => *val,
        }
    }

    pub fn cast_to(&self, other: impl Into<Type>) -> ConstantValue {
        match other.into().storage_type() {
            StorageType::Scalar(elem_type) => match elem_type {
                ElemType::Float(kind) => match kind {
                    FloatKind::E2M1 => e2m1::from_f64(self.as_f64()).to_f64(),
                    FloatKind::E2M3 | FloatKind::E3M2 => {
                        unimplemented!("FP6 constants not yet supported")
                    }
                    FloatKind::E4M3 => e4m3::from_f64(self.as_f64()).to_f64(),
                    FloatKind::E5M2 => e5m2::from_f64(self.as_f64()).to_f64(),
                    FloatKind::UE8M0 => ue8m0::from_f64(self.as_f64()).to_f64(),
                    FloatKind::F16 => half::f16::from_f64(self.as_f64()).to_f64(),
                    FloatKind::BF16 => half::bf16::from_f64(self.as_f64()).to_f64(),
                    FloatKind::Flex32 | FloatKind::TF32 | FloatKind::F32 => {
                        self.as_f64() as f32 as f64
                    }
                    FloatKind::F64 => self.as_f64(),
                }
                .into(),
                ElemType::Int(kind) => match kind {
                    IntKind::I8 => self.as_i64() as i8 as i64,
                    IntKind::I16 => self.as_i64() as i16 as i64,
                    IntKind::I32 => self.as_i64() as i32 as i64,
                    IntKind::I64 => self.as_i64(),
                }
                .into(),
                ElemType::UInt(kind) => match kind {
                    UIntKind::U8 => self.as_u64() as u8 as u64,
                    UIntKind::U16 => self.as_u64() as u16 as u64,
                    UIntKind::U32 => self.as_u64() as u32 as u64,
                    UIntKind::U64 => self.as_u64(),
                }
                .into(),
                ElemType::Bool => self.as_bool().into(),
            },
            StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2) => {
                e2m1::from_f64(self.as_f64()).to_f64().into()
            }
            StorageType::Packed(..) => unimplemented!("Unsupported packed type"),
        }
    }
}

impl Display for ConstantValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConstantValue::Int(val) => write!(f, "{val}"),
            ConstantValue::Float(val) => write!(f, "{val:?}"),
            ConstantValue::UInt(val) => write!(f, "{val}"),
            ConstantValue::Bool(val) => write!(f, "{val}"),
        }
    }
}

impl ExpandValue {
    pub fn as_const(&self) -> Option<ConstantValue> {
        match self {
            ExpandValue::Constant { value, .. } => Some(*value),
            _ => None,
        }
    }
}

impl Display for ExpandValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExpandValue::Constant { value, ty } => write!(f, "{ty}({value})"),
            ExpandValue::Value(value) => write!(f, "{value:?}"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&ExpandValue> for ExpandValue {
    fn from(value: &ExpandValue) -> Self {
        *value
    }
}
