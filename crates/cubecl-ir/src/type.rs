use super::{ConstantScalarValue, Variable, VariableKind};
use crate::{BarrierLevel, TypeHash};
use core::fmt::Display;
use cubecl_common::{
    e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32,
    quant::scheme::{QuantParam, QuantValue},
    tf32, ue8m0,
};

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
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    Packed(ElemType, usize),
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
    /// Create a constant scalar from a float.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_f64(&self, val: f64) -> Variable {
        Variable::constant(match self {
            ElemType::Float(kind) => ConstantScalarValue::Float(val, *kind),
            ElemType::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            ElemType::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            ElemType::Bool => ConstantScalarValue::Bool(val > 0.0),
        })
    }
    /// Create a constant scalar from a signed integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_i64(&self, val: i64) -> Variable {
        Variable::constant(match self {
            ElemType::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            ElemType::Int(kind) => ConstantScalarValue::Int(val, *kind),
            ElemType::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            ElemType::Bool => ConstantScalarValue::Bool(val > 0),
        })
    }
    /// Create a constant scalar from a unsigned integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_u64(&self, val: u64) -> Variable {
        Variable::constant(match self {
            ElemType::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            ElemType::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            ElemType::UInt(kind) => ConstantScalarValue::UInt(val, *kind),
            ElemType::Bool => ConstantScalarValue::Bool(val > 0),
        })
    }
    /// Create a constant scalar from a boolean.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_bool(&self, val: bool) -> Variable {
        Variable::constant(match self {
            ElemType::Float(kind) => ConstantScalarValue::Float(val as u32 as f64, *kind),
            ElemType::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            ElemType::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            ElemType::Bool => ConstantScalarValue::Bool(val),
        })
    }

    /// Ensure that the variable provided, when a constant, is the same type as elem.
    pub fn from_constant(&self, constant: Variable) -> Variable {
        let value = match constant.kind {
            VariableKind::ConstantScalar(value) => value,
            _ => return constant,
        };

        match value {
            ConstantScalarValue::Int(val, _) => self.constant_from_i64(val),
            ConstantScalarValue::Float(val, _) => self.constant_from_f64(val),
            ConstantScalarValue::UInt(val, _) => self.constant_from_u64(val),
            ConstantScalarValue::Bool(val) => self.constant_from_bool(val),
        }
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

    pub fn max_variable(&self) -> Variable {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => ConstantScalarValue::Float(e2m1::MAX, FloatKind::E2M1),
                FloatKind::E2M3 => ConstantScalarValue::Float(e2m3::MAX, FloatKind::E2M3),
                FloatKind::E3M2 => ConstantScalarValue::Float(e3m2::MAX, FloatKind::E3M2),
                FloatKind::E4M3 => ConstantScalarValue::Float(e4m3::MAX, FloatKind::E4M3),
                FloatKind::E5M2 => ConstantScalarValue::Float(e5m2::MAX, FloatKind::E5M2),
                FloatKind::UE8M0 => ConstantScalarValue::Float(ue8m0::MAX, FloatKind::UE8M0),
                FloatKind::F16 => {
                    ConstantScalarValue::Float(half::f16::MAX.to_f64(), FloatKind::F16)
                }
                FloatKind::BF16 => {
                    ConstantScalarValue::Float(half::bf16::MAX.to_f64(), FloatKind::BF16)
                }
                FloatKind::Flex32 => ConstantScalarValue::Float(f32::MAX.into(), FloatKind::Flex32),
                FloatKind::F32 => ConstantScalarValue::Float(f32::MAX.into(), FloatKind::F32),
                FloatKind::TF32 => ConstantScalarValue::Float(f32::MAX.into(), FloatKind::TF32),
                FloatKind::F64 => ConstantScalarValue::Float(f64::MAX, FloatKind::F64),
            },
            ElemType::Int(kind) => match kind {
                IntKind::I8 => ConstantScalarValue::Int(i8::MAX.into(), IntKind::I8),
                IntKind::I16 => ConstantScalarValue::Int(i16::MAX.into(), IntKind::I16),
                IntKind::I32 => ConstantScalarValue::Int(i32::MAX.into(), IntKind::I32),
                IntKind::I64 => ConstantScalarValue::Int(i64::MAX, IntKind::I64),
            },
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => ConstantScalarValue::UInt(u8::MAX.into(), UIntKind::U8),
                UIntKind::U16 => ConstantScalarValue::UInt(u16::MAX.into(), UIntKind::U16),
                UIntKind::U32 => ConstantScalarValue::UInt(u32::MAX.into(), UIntKind::U32),
                UIntKind::U64 => ConstantScalarValue::UInt(u64::MAX, UIntKind::U64),
            },
            ElemType::Bool => ConstantScalarValue::Bool(true),
        };

        Variable::new(VariableKind::ConstantScalar(value), Type::scalar(*self))
    }

    pub fn min_variable(&self) -> Variable {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => ConstantScalarValue::Float(e2m1::MIN, FloatKind::E2M1),
                FloatKind::E2M3 => ConstantScalarValue::Float(e2m3::MIN, FloatKind::E2M3),
                FloatKind::E3M2 => ConstantScalarValue::Float(e3m2::MIN, FloatKind::E3M2),
                FloatKind::E4M3 => ConstantScalarValue::Float(e4m3::MIN, FloatKind::E4M3),
                FloatKind::E5M2 => ConstantScalarValue::Float(e5m2::MIN, FloatKind::E5M2),
                FloatKind::UE8M0 => ConstantScalarValue::Float(ue8m0::MIN, FloatKind::UE8M0),
                FloatKind::F16 => {
                    ConstantScalarValue::Float(half::f16::MIN.to_f64(), FloatKind::F16)
                }
                FloatKind::BF16 => {
                    ConstantScalarValue::Float(half::bf16::MIN.to_f64(), FloatKind::BF16)
                }
                FloatKind::Flex32 => ConstantScalarValue::Float(f32::MIN.into(), FloatKind::Flex32),
                FloatKind::F32 => ConstantScalarValue::Float(f32::MIN.into(), FloatKind::F32),
                FloatKind::TF32 => ConstantScalarValue::Float(f32::MIN.into(), FloatKind::TF32),
                FloatKind::F64 => ConstantScalarValue::Float(f64::MIN, FloatKind::F64),
            },
            ElemType::Int(kind) => match kind {
                IntKind::I8 => ConstantScalarValue::Int(i8::MIN.into(), IntKind::I8),
                IntKind::I16 => ConstantScalarValue::Int(i16::MIN.into(), IntKind::I16),
                IntKind::I32 => ConstantScalarValue::Int(i32::MIN.into(), IntKind::I32),
                IntKind::I64 => ConstantScalarValue::Int(i64::MIN, IntKind::I64),
            },
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => ConstantScalarValue::UInt(u8::MIN.into(), UIntKind::U8),
                UIntKind::U16 => ConstantScalarValue::UInt(u16::MIN.into(), UIntKind::U16),
                UIntKind::U32 => ConstantScalarValue::UInt(u32::MIN.into(), UIntKind::U32),
                UIntKind::U64 => ConstantScalarValue::UInt(u64::MIN, UIntKind::U64),
            },
            ElemType::Bool => ConstantScalarValue::Bool(false),
        };

        Variable::new(VariableKind::ConstantScalar(value), Type::scalar(*self))
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

    pub fn packing_factor(&self) -> usize {
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
            StorageType::Packed(ty, factor) => ty.size_bits() * *factor,
            StorageType::Scalar(ty) | StorageType::Atomic(ty) => ty.size_bits(),
            StorageType::Opaque(ty) => ty.size_bits(),
        }
    }

    /// Ensure that the variable provided, when a constant, is the same type as elem.
    pub fn from_constant(&self, constant: Variable) -> Variable {
        self.elem_type().from_constant(constant)
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
}

impl From<ElemType> for StorageType {
    fn from(val: ElemType) -> Self {
        StorageType::Scalar(val)
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
    Line(StorageType, LineSize),
    /// No defined physical representation, purely semantic. i.e. barrier, pipeline
    Semantic(SemanticType),
}

pub type LineSize = usize;

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

    pub fn line_size(&self) -> LineSize {
        match self {
            Type::Scalar(_) => 1,
            Type::Line(_, line_size) => *line_size,
            Type::Semantic(_) => 0,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size(),
            Type::Line(ty, line_size) => ty.size() * *line_size,
            Type::Semantic(_) => 0,
        }
    }

    pub fn size_bits(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size_bits(),
            Type::Line(ty, line_size) => ty.size_bits() * *line_size,
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
        Variable::constant(ConstantScalarValue::Bool(value))
    }
}

impl From<i8> for Variable {
    fn from(value: i8) -> Self {
        Variable::constant(ConstantScalarValue::Int(value as i64, IntKind::I8))
    }
}

impl From<i16> for Variable {
    fn from(value: i16) -> Self {
        Variable::constant(ConstantScalarValue::Int(value as i64, IntKind::I16))
    }
}

impl From<i32> for Variable {
    fn from(value: i32) -> Self {
        Variable::constant(ConstantScalarValue::Int(value as i64, IntKind::I32))
    }
}

impl From<i64> for Variable {
    fn from(value: i64) -> Self {
        Variable::constant(ConstantScalarValue::Int(value, IntKind::I64))
    }
}

impl From<e2m1> for Variable {
    fn from(_value: e2m1) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<e2m1x2> for Variable {
    fn from(_value: e2m1x2) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<e2m3> for Variable {
    fn from(_value: e2m3) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<e3m2> for Variable {
    fn from(_value: e3m2) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<e4m3> for Variable {
    fn from(_value: e4m3) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<e5m2> for Variable {
    fn from(_value: e5m2) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<ue8m0> for Variable {
    fn from(_value: ue8m0) -> Self {
        unimplemented!("Can't currently construct minifloats")
    }
}

impl From<half::f16> for Variable {
    fn from(value: half::f16) -> Self {
        Variable::constant(ConstantScalarValue::Float(value.to_f64(), FloatKind::F16))
    }
}

impl From<half::bf16> for Variable {
    fn from(value: half::bf16) -> Self {
        Variable::constant(ConstantScalarValue::Float(value.to_f64(), FloatKind::BF16))
    }
}

impl From<flex32> for Variable {
    fn from(value: flex32) -> Self {
        Variable::constant(ConstantScalarValue::Float(
            value.to_f64(),
            FloatKind::Flex32,
        ))
    }
}

impl From<tf32> for Variable {
    fn from(value: tf32) -> Self {
        Variable::constant(ConstantScalarValue::Float(value.to_f64(), FloatKind::TF32))
    }
}

impl From<f32> for Variable {
    fn from(value: f32) -> Self {
        Variable::constant(ConstantScalarValue::Float(value as f64, FloatKind::F32))
    }
}

impl From<f64> for Variable {
    fn from(value: f64) -> Self {
        Variable::constant(ConstantScalarValue::Float(value, FloatKind::F64))
    }
}

impl From<u8> for Variable {
    fn from(value: u8) -> Self {
        Variable::constant(ConstantScalarValue::UInt(value as u64, UIntKind::U8))
    }
}

impl From<u16> for Variable {
    fn from(value: u16) -> Self {
        Variable::constant(ConstantScalarValue::UInt(value as u64, UIntKind::U16))
    }
}

impl From<u32> for Variable {
    fn from(value: u32) -> Self {
        Variable::constant(ConstantScalarValue::UInt(value as u64, UIntKind::U32))
    }
}

impl From<u64> for Variable {
    fn from(value: u64) -> Self {
        Variable::constant(ConstantScalarValue::UInt(value, UIntKind::U64))
    }
}

impl From<usize> for Variable {
    fn from(value: usize) -> Self {
        Variable::constant(ConstantScalarValue::UInt(value as u64, UIntKind::U32))
    }
}
