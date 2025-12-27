use super::{ConstantValue, Variable, VariableKind};
use crate::{BarrierLevel, TypeHash};
use core::fmt::Display;
use cubecl_common::{
    e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32,
    quant::scheme::{QuantParam, QuantValue},
    tf32, ue8m0,
};
use derive_more::From;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum FloatKind {
    /// FP4, 2 bit exponent, 1 bit mantissa
    E2M1,
    /// FP6, 2 bit exponent, 3 bit mantissa
    /// Note: represented by an 8-bit value, with the upper two bits being insignificant
    E2M3,
    /// FP6, 3 bit exponent, 2 bit mantissa
    /// Note: represented by an 8-bit value, with the upper two bits being insignificant
    E3M2,
    /// FP8, 4 bit exponent, 3 bit mantissa
    E4M3,
    /// FP8, 5 bit exponent, 2 bit mantissa
    E5M2,
    /// FP8, unsigned, 8 bit exponent, 0 bit mantissa
    UE8M0,
    F16,
    BF16,
    Flex32,
    F32,
    TF32,
    F64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum UIntKind {
    U8,
    U16,
    U32,
    U64,
}

/// Conceptual element type, not necessarily the physical type used in the code
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, From)]
#[allow(missing_docs)]
pub enum ElemType {
    Float(FloatKind),
    Int(IntKind),
    UInt(UIntKind),
    Bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpaqueType {
    Barrier(BarrierLevel),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemanticType {
    BarrierToken,
    Pipeline,
    TensorMap,
}

/// Physical type containing one or more elements
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StorageType {
    /// `ElemType` is the same as the physical type
    Scalar(ElemType),
    /// Packed values of type `ElemType`
    Packed(ElemType, u32),
    /// Atomically accessed version of `ElemType`
    Atomic(ElemType),
    /// Opaque types that can be stored but not interacted with normally. Currently only barrier,
    /// but may be used for arrival tokens and tensor map descriptors, for example.
    Opaque(OpaqueType),
}

impl ElemType {
    /// Creates an elem type that correspond to the given [QuantParam].
    pub fn from_quant_param(quant_param: QuantParam) -> Self {
        match quant_param {
            QuantParam::F32 => Self::Float(FloatKind::F32),
            QuantParam::F16 => Self::Float(FloatKind::F16),
            QuantParam::BF16 => Self::Float(FloatKind::BF16),
            QuantParam::UE8M0 => Self::Float(FloatKind::UE8M0),
            QuantParam::UE4M3 => Self::Float(FloatKind::UE8M0),
        }
    }

    /// Creates an elem type that correspond to the given [QuantValue].
    pub fn from_quant_value(quant_value: QuantValue) -> Self {
        match quant_value {
            QuantValue::E5M2 => Self::Float(FloatKind::E5M2),
            QuantValue::E4M3 => Self::Float(FloatKind::E4M3),
            QuantValue::E2M1 => Self::Float(FloatKind::E2M1),
            QuantValue::Q8F | QuantValue::Q8S => Self::Int(IntKind::I8),
            other => panic!("Unsupported quant value {other:?}"),
        }
    }

    /// Create a constant from a constant value.
    ///
    /// The output will have the same type as the element.
    pub fn constant(&self, val: ConstantValue) -> Variable {
        Variable::constant(val, Type::scalar(*self))
    }

    /// Get the size in bytes.
    pub const fn size(&self) -> usize {
        match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0 => core::mem::size_of::<u8>(),
                FloatKind::F16 => core::mem::size_of::<half::f16>(),
                FloatKind::BF16 => core::mem::size_of::<half::bf16>(),
                FloatKind::F32 => core::mem::size_of::<f32>(),
                FloatKind::F64 => core::mem::size_of::<f64>(),
                FloatKind::Flex32 => core::mem::size_of::<f32>(),
                FloatKind::TF32 => core::mem::size_of::<f32>(),
            },
            ElemType::Int(kind) => match kind {
                IntKind::I8 => core::mem::size_of::<i8>(),
                IntKind::I16 => core::mem::size_of::<i16>(),
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => core::mem::size_of::<u8>(),
                UIntKind::U16 => core::mem::size_of::<u16>(),
                UIntKind::U32 => core::mem::size_of::<u32>(),
                UIntKind::U64 => core::mem::size_of::<u64>(),
            },
            ElemType::Bool => core::mem::size_of::<bool>(),
        }
    }

    /// Get the size in bits.
    pub const fn size_bits(&self) -> usize {
        match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0
                | FloatKind::F16
                | FloatKind::BF16
                | FloatKind::F32
                | FloatKind::F64
                | FloatKind::Flex32
                | FloatKind::TF32 => self.size() * 8,
                FloatKind::E2M1 => 4,
            },
            ElemType::Int(_) | ElemType::UInt(_) | ElemType::Bool => self.size() * 8,
        }
    }

    pub const fn min_line_size(&self) -> u8 {
        match self {
            ElemType::Float(FloatKind::E2M1) => 2,
            _ => 1,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, ElemType::Int(_) | ElemType::UInt(_) | ElemType::Bool)
    }

    pub fn is_signed_int(&self) -> bool {
        matches!(self, ElemType::Int(_))
    }

    pub fn is_unsigned_int(&self) -> bool {
        matches!(self, ElemType::UInt(_) | ElemType::Bool)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ElemType::Float(_))
    }

    pub fn as_float(&self) -> Option<FloatKind> {
        match self {
            ElemType::Float(kind) => Some(*kind),
            _ => None,
        }
    }

    pub fn max_variable(&self) -> Variable {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MAX,
                FloatKind::E2M3 => e2m3::MAX,
                FloatKind::E3M2 => e3m2::MAX,
                FloatKind::E4M3 => e4m3::MAX,
                FloatKind::E5M2 => e5m2::MAX,
                FloatKind::UE8M0 => ue8m0::MAX,
                FloatKind::F16 => half::f16::MAX.to_f64(),
                FloatKind::BF16 => half::bf16::MAX.to_f64(),
                FloatKind::Flex32 | FloatKind::TF32 | FloatKind::F32 => f32::MAX as f64,
                FloatKind::F64 => f64::MAX,
            }
            .into(),
            ElemType::Int(kind) => match kind {
                IntKind::I8 => i8::MAX as i64,
                IntKind::I16 => i16::MAX as i64,
                IntKind::I32 => i32::MAX as i64,
                IntKind::I64 => i64::MAX,
            }
            .into(),
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => u8::MAX as u64,
                UIntKind::U16 => u16::MAX as u64,
                UIntKind::U32 => u32::MAX as u64,
                UIntKind::U64 => u64::MAX,
            }
            .into(),
            ElemType::Bool => true.into(),
        };

        Variable::new(VariableKind::Constant(value), Type::scalar(*self))
    }

    pub fn min_variable(&self) -> Variable {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MIN,
                FloatKind::E2M3 => e2m3::MIN,
                FloatKind::E3M2 => e3m2::MIN,
                FloatKind::E4M3 => e4m3::MIN,
                FloatKind::E5M2 => e5m2::MIN,
                FloatKind::UE8M0 => ue8m0::MIN,
                FloatKind::F16 => half::f16::MIN.to_f64(),
                FloatKind::BF16 => half::bf16::MIN.to_f64(),
                FloatKind::Flex32 | FloatKind::TF32 | FloatKind::F32 => f32::MIN as f64,
                FloatKind::F64 => f64::MIN,
            }
            .into(),
            ElemType::Int(kind) => match kind {
                IntKind::I8 => i8::MIN as i64,
                IntKind::I16 => i16::MIN as i64,
                IntKind::I32 => i32::MIN as i64,
                IntKind::I64 => i64::MIN,
            }
            .into(),
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => u8::MIN as u64,
                UIntKind::U16 => u16::MIN as u64,
                UIntKind::U32 => u32::MIN as u64,
                UIntKind::U64 => u64::MIN,
            }
            .into(),
            ElemType::Bool => false.into(),
        };

        Variable::new(VariableKind::Constant(value), Type::scalar(*self))
    }

    pub fn epsilon(&self) -> f64 {
        match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => 0.5 * (e2m1::MAX - e2m1::MIN),
                FloatKind::E2M3 => 0.5 * (e2m3::MAX - e2m3::MIN),
                FloatKind::E3M2 => 0.5 * (e3m2::MAX - e3m2::MIN),
                FloatKind::E4M3 => 0.5 * (e4m3::MAX - e4m3::MIN),
                FloatKind::E5M2 => 0.5 * (e5m2::MAX - e5m2::MIN),
                FloatKind::UE8M0 => 0.5 * (ue8m0::MAX - ue8m0::MIN),
                FloatKind::F16 => half::f16::EPSILON.to_f64(),
                FloatKind::BF16 => 0.0078125, // bf16 epsilon ≈ 2^-7
                FloatKind::Flex32 | FloatKind::F32 | FloatKind::TF32 => f32::EPSILON.into(),
                FloatKind::F64 => f64::EPSILON,
            },
            ElemType::Int(_) | ElemType::UInt(_) => 1.0, // step of 1
            ElemType::Bool => 1.0,
        }
    }
}

impl OpaqueType {
    /// Get the size in bytes.
    pub const fn size(&self) -> usize {
        match self {
            OpaqueType::Barrier(_) => 8,
        }
    }

    /// Get the size in bits.
    pub const fn size_bits(&self) -> usize {
        match self {
            OpaqueType::Barrier(_) => 64,
        }
    }
}

impl StorageType {
    pub fn elem_type(&self) -> ElemType {
        match self {
            StorageType::Scalar(ty) | StorageType::Packed(ty, _) | StorageType::Atomic(ty) => *ty,
            StorageType::Opaque(_) => unimplemented!("Can't get elem type for opaque type"),
        }
    }

    pub fn packing_factor(&self) -> u32 {
        match self {
            StorageType::Packed(_, factor) => *factor,
            _ => 1,
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, StorageType::Atomic(_))
    }

    pub fn size(&self) -> usize {
        self.size_bits().div_ceil(8)
    }

    pub fn size_bits(&self) -> usize {
        match self {
            StorageType::Packed(ty, factor) => ty.size_bits() * *factor as usize,
            StorageType::Scalar(ty) | StorageType::Atomic(ty) => ty.size_bits(),
            StorageType::Opaque(ty) => ty.size_bits(),
        }
    }

    pub fn is_int(&self) -> bool {
        self.elem_type().is_int()
    }

    pub fn is_signed_int(&self) -> bool {
        self.elem_type().is_signed_int()
    }

    pub fn is_unsigned_int(&self) -> bool {
        self.elem_type().is_unsigned_int()
    }

    pub fn is_float(&self) -> bool {
        self.elem_type().is_float()
    }

    /// Returns an empirical epsilon for this storage type, taking quantization into account.
    pub fn epsilon(&self) -> f64 {
        match self {
            StorageType::Scalar(ty) | StorageType::Atomic(ty) => ty.epsilon(),
            StorageType::Packed(ty, factor) => {
                // For packed types, we can conservatively scale epsilon by the number of packed elements
                ty.epsilon() * (*factor as f64)
            }
            StorageType::Opaque(_) => panic!("Opaque type does not have an epsilon"),
        }
    }

    pub fn constant(&self, value: ConstantValue) -> Variable {
        Variable::constant(value, Type::new(*self))
    }
}

impl<T: Into<ElemType>> From<T> for StorageType {
    fn from(val: T) -> Self {
        StorageType::Scalar(val.into())
    }
}

impl From<OpaqueType> for StorageType {
    fn from(val: OpaqueType) -> Self {
        StorageType::Opaque(val)
    }
}

impl<T: Into<StorageType>> From<T> for Type {
    fn from(val: T) -> Self {
        Type::new(val.into())
    }
}

impl From<SemanticType> for Type {
    fn from(val: SemanticType) -> Self {
        Type::semantic(val)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    /// Scalar type containing a single storage element
    Scalar(StorageType),
    /// Line wrapping `n` storage elements
    Line(StorageType, u32),
    /// No defined physical representation, purely semantic. i.e. barrier, pipeline
    Semantic(SemanticType),
}

pub type LineSize = u32;

impl Type {
    /// Fetch the elem of the item.
    pub fn elem_type(&self) -> ElemType {
        self.storage_type().elem_type()
    }

    /// Create a new item
    pub fn new(storage: StorageType) -> Self {
        Type::Scalar(storage)
    }

    pub fn scalar(elem: ElemType) -> Self {
        Self::new(StorageType::Scalar(elem))
    }

    pub fn semantic(ty: SemanticType) -> Self {
        Self::Semantic(ty)
    }

    pub fn line(self, line_size: LineSize) -> Type {
        match line_size > 1 {
            true => Type::Line(self.storage_type(), line_size),
            false => Type::Scalar(self.storage_type()),
        }
    }

    pub fn line_size(&self) -> u32 {
        match self {
            Type::Scalar(_) => 1,
            Type::Line(_, line_size) => *line_size,
            Type::Semantic(_) => 0,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size(),
            Type::Line(ty, line_size) => ty.size() * *line_size as usize,
            Type::Semantic(_) => 0,
        }
    }

    pub fn size_bits(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size_bits(),
            Type::Line(ty, line_size) => ty.size_bits() * *line_size as usize,
            Type::Semantic(_) => 0,
        }
    }

    pub fn is_atomic(&self) -> bool {
        !self.is_semantic() && self.storage_type().is_atomic()
    }

    pub fn is_int(&self) -> bool {
        !self.is_semantic() && self.storage_type().is_int()
    }

    pub fn is_signed_int(&self) -> bool {
        !self.is_semantic() && self.storage_type().is_signed_int()
    }

    pub fn is_unsigned_int(&self) -> bool {
        !self.is_semantic() && self.storage_type().is_unsigned_int()
    }

    pub fn is_float(&self) -> bool {
        !self.is_semantic() && self.storage_type().is_float()
    }

    pub fn storage_type(&self) -> StorageType {
        match self {
            Type::Scalar(ty) | Type::Line(ty, _) => *ty,
            Type::Semantic(_) => unimplemented!("Can't get storage for semantic type"),
        }
    }

    pub fn is_semantic(&self) -> bool {
        match self {
            Type::Scalar(_) | Type::Line(_, _) => false,
            Type::Semantic(_) => true,
        }
    }

    pub fn constant(&self, value: ConstantValue) -> Variable {
        Variable::constant(value, *self)
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Type::Scalar(ty) => write!(f, "{ty}"),
            Type::Line(ty, line_size) => write!(f, "line<{ty}, {line_size}>"),
            Type::Semantic(ty) => write!(f, "{ty}"),
        }
    }
}

impl Display for StorageType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StorageType::Scalar(ty) => write!(f, "{ty}"),
            StorageType::Packed(ty, factor) => write!(f, "packed<{ty}, {factor}>"),
            StorageType::Atomic(ty) => write!(f, "atomic<{ty}>"),
            StorageType::Opaque(ty) => write!(f, "{ty}"),
        }
    }
}

impl Display for ElemType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float(kind) => match kind {
                FloatKind::E2M1 => f.write_str("e2m1"),
                FloatKind::E2M3 => f.write_str("e2m3"),
                FloatKind::E3M2 => f.write_str("e3m2"),
                FloatKind::E4M3 => f.write_str("e4m3"),
                FloatKind::E5M2 => f.write_str("e5m2"),
                FloatKind::UE8M0 => f.write_str("ue8m0"),
                FloatKind::F16 => f.write_str("f16"),
                FloatKind::BF16 => f.write_str("bf16"),
                FloatKind::Flex32 => f.write_str("flex32"),
                FloatKind::TF32 => f.write_str("tf32"),
                FloatKind::F32 => f.write_str("f32"),
                FloatKind::F64 => f.write_str("f64"),
            },
            Self::Int(kind) => match kind {
                IntKind::I8 => f.write_str("i8"),
                IntKind::I16 => f.write_str("i16"),
                IntKind::I32 => f.write_str("i32"),
                IntKind::I64 => f.write_str("i64"),
            },
            Self::UInt(kind) => match kind {
                UIntKind::U8 => f.write_str("u8"),
                UIntKind::U16 => f.write_str("u16"),
                UIntKind::U32 => f.write_str("u32"),
                UIntKind::U64 => f.write_str("u64"),
            },
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for SemanticType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SemanticType::BarrierToken => f.write_str("barrier_token"),
            SemanticType::Pipeline => f.write_str("pipeline"),
            SemanticType::TensorMap => f.write_str("tensor_map"),
        }
    }
}

impl Display for OpaqueType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OpaqueType::Barrier(level) => write!(f, "barrier<{level}>"),
        }
    }
}

impl From<bool> for Variable {
    fn from(value: bool) -> Self {
        Variable::constant(ConstantValue::Bool(value), ElemType::Bool)
    }
}

impl From<i8> for Variable {
    fn from(value: i8) -> Self {
        Variable::constant(ConstantValue::Int(value as i64), IntKind::I8)
    }
}

impl From<i16> for Variable {
    fn from(value: i16) -> Self {
        Variable::constant(ConstantValue::Int(value as i64), IntKind::I16)
    }
}

impl From<i32> for Variable {
    fn from(value: i32) -> Self {
        Variable::constant(ConstantValue::Int(value as i64), IntKind::I32)
    }
}

impl From<i64> for Variable {
    fn from(value: i64) -> Self {
        Variable::constant(ConstantValue::Int(value), IntKind::I64)
    }
}

impl From<e2m1> for Variable {
    fn from(value: e2m1) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::E2M1)
    }
}

impl From<e2m1x2> for Variable {
    fn from(_value: e2m1x2) -> Self {
        unimplemented!("Can't currently construct e2m1x2")
    }
}

impl From<e2m3> for Variable {
    fn from(_value: e2m3) -> Self {
        unimplemented!("Can't currently construct fp6")
    }
}

impl From<e3m2> for Variable {
    fn from(_value: e3m2) -> Self {
        unimplemented!("Can't currently construct fp6")
    }
}

impl From<e4m3> for Variable {
    fn from(value: e4m3) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::E4M3)
    }
}

impl From<e5m2> for Variable {
    fn from(value: e5m2) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::E5M2)
    }
}

impl From<ue8m0> for Variable {
    fn from(value: ue8m0) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::UE8M0)
    }
}

impl From<half::f16> for Variable {
    fn from(value: half::f16) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::F16)
    }
}

impl From<half::bf16> for Variable {
    fn from(value: half::bf16) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::BF16)
    }
}

impl From<flex32> for Variable {
    fn from(value: flex32) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::Flex32)
    }
}

impl From<tf32> for Variable {
    fn from(value: tf32) -> Self {
        Variable::constant(ConstantValue::Float(value.to_f64()), FloatKind::TF32)
    }
}

impl From<f32> for Variable {
    fn from(value: f32) -> Self {
        Variable::constant(ConstantValue::Float(value as f64), FloatKind::F32)
    }
}

impl From<f64> for Variable {
    fn from(value: f64) -> Self {
        Variable::constant(ConstantValue::Float(value), FloatKind::F64)
    }
}

impl From<u8> for Variable {
    fn from(value: u8) -> Self {
        Variable::constant(ConstantValue::UInt(value as u64), UIntKind::U8)
    }
}

impl From<u16> for Variable {
    fn from(value: u16) -> Self {
        Variable::constant(ConstantValue::UInt(value as u64), UIntKind::U16)
    }
}

impl From<u32> for Variable {
    fn from(value: u32) -> Self {
        Variable::constant(ConstantValue::UInt(value as u64), UIntKind::U32)
    }
}

impl From<u64> for Variable {
    fn from(value: u64) -> Self {
        Variable::constant(ConstantValue::UInt(value), UIntKind::U64)
    }
}

impl From<usize> for Variable {
    fn from(value: usize) -> Self {
        Variable::constant(ConstantValue::UInt(value as u64), UIntKind::U32)
    }
}
