use core::{fmt::Display, hash::Hash};

use crate::{AddressSpace, FloatKind, IntKind, StorageType, TypeHash};

use super::{ElemType, Type, UIntKind};
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
use derive_more::From;
use float_ord::FloatOrd;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_docs)]
pub struct Value {
    pub kind: ValueKind,
    pub ty: Type,
}

impl Value {
    pub fn new(id: Id, ty: Type) -> Self {
        Self {
            kind: ValueKind::Value { id },
            ty,
        }
    }

    pub fn constant(value: ConstantValue, ty: impl Into<Type>) -> Self {
        let ty = ty.into();
        let value = value.cast_to(ty);
        Self {
            kind: ValueKind::Constant(value),
            ty,
        }
    }

    pub fn elem_type(&self) -> ElemType {
        self.ty.elem_type()
    }

    pub fn storage_type(&self) -> StorageType {
        self.ty.storage_type()
    }

    pub fn can_mutate(&self) -> bool {
        self.ty.is_ptr()
    }

    pub fn address_space(&self) -> AddressSpace {
        match self.ty {
            Type::Pointer(_, addr_space) => addr_space,
            _ => match self.kind {
                ValueKind::Value { .. } | ValueKind::Constant(..) => AddressSpace::Local,
            },
        }
    }

    pub fn value_type(&self) -> Type {
        self.ty.value_type()
    }
}

pub type Id = u32;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueKind {
    Value { id: Id },
    Constant(ConstantValue),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord)]
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

impl Value {
    /// Whether a value is always immutable. Used for optimizations to determine whether it's
    /// safe to inline/merge
    pub fn is_immutable(&self) -> bool {
        !self.can_mutate()
    }

    /// Is this an array type that yields items when indexed,
    /// or a scalar/vector that yields elems/slices when indexed?
    pub fn is_array_like(&self) -> bool {
        self.ty.is_array_like()
    }

    pub fn is_value(&self) -> bool {
        self.ty.is_value()
    }

    /// Is this an value type that is contained in concrete memory,
    /// or a local array/scalar/vector?
    pub fn is_memory(&self) -> bool {
        matches!(
            self.address_space(),
            AddressSpace::Global(_) | AddressSpace::Shared
        )
    }

    pub fn has_buffer_length(&self) -> bool {
        matches!(self.address_space(), AddressSpace::Global(_))
    }

    /// Determines if the value is a constant with the specified value (converted if necessary)
    pub fn is_constant(&self, value: i64) -> bool {
        match self.kind {
            ValueKind::Constant(ConstantValue::Int(val)) => val == value,
            ValueKind::Constant(ConstantValue::UInt(val)) => val as i64 == value,
            ValueKind::Constant(ConstantValue::Float(val)) => val == value as f64,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `true` value
    pub fn is_true(&self) -> bool {
        match self.kind {
            ValueKind::Constant(ConstantValue::Bool(val)) => val,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `false` value
    pub fn is_false(&self) -> bool {
        match self.kind {
            ValueKind::Constant(ConstantValue::Bool(val)) => !val,
            _ => false,
        }
    }
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

impl Value {
    pub fn vector_size(&self) -> usize {
        self.ty.vector_size()
    }

    pub fn id(&self) -> Id {
        match self.kind {
            ValueKind::Value { id, .. } => id,
            _ => panic!(),
        }
    }

    pub fn as_const(&self) -> Option<ConstantValue> {
        match self.kind {
            ValueKind::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            ValueKind::Constant(constant) => write!(f, "{}({constant})", self.ty),
            other => write!(f, "{other}"),
        }
    }
}

impl Display for ValueKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ValueKind::Constant(constant) => write!(f, "{constant}"),
            ValueKind::Value { id } => write!(f, "%{id}"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Value> for Value {
    fn from(value: &Value) -> Self {
        *value
    }
}
