use super::{ConstantValue, Value, ValueKind};
use crate::{BarrierLevel, ClampMode, EnumCounts, Id, MatrixType, TypeHash};
use core::fmt::Display;
use cubecl_common::{
    e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32,
    quant::scheme::{QuantParam, QuantValue},
    tf32, ue8m0,
};
use derive_more::{Display, From};
use half::{bf16, f16};

pub use internment::Intern;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, EnumCounts)]
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
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, EnumCounts)]
#[allow(missing_docs)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, EnumCounts)]
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
    BarrierToken(BarrierLevel),
    TensorMap,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemanticType {
    TensorLayout(usize, ClampMode),
    TensorView(usize, bool, [u32; 5]),
}

/// Physical type containing one or more elements
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StorageType {
    /// `ElemType` is the same as the physical type
    Scalar(ElemType),
    /// Packed values of type `ElemType`
    Packed(ElemType, usize),
}

impl core::fmt::Debug for StorageType {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        // Ensure debug is not spread into multiple lines because it makes kernel ids very hard
        // to read.
        struct Dummy<'a>(&'a StorageType);

        impl<'a> core::fmt::Debug for Dummy<'a> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                match self.0 {
                    StorageType::Scalar(f0) => f.debug_tuple("Scalar").field(&f0).finish(),
                    StorageType::Packed(f0, f1) => {
                        f.debug_tuple("Packed").field(&f0).field(&f1).finish()
                    }
                }
            }
        }

        write!(f, "{:?}", Dummy(self))
    }
}

impl ElemType {
    /// Creates an elem type that correspond to the given [`QuantParam`].
    pub fn from_quant_param(quant_param: QuantParam) -> Self {
        match quant_param {
            QuantParam::F32 => Self::Float(FloatKind::F32),
            QuantParam::F16 => Self::Float(FloatKind::F16),
            QuantParam::BF16 => Self::Float(FloatKind::BF16),
            QuantParam::UE8M0 => Self::Float(FloatKind::UE8M0),
            QuantParam::UE4M3 => Self::Float(FloatKind::UE8M0),
        }
    }

    /// Creates an elem type that correspond to the given [`QuantValue`].
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
    pub fn constant(&self, val: ConstantValue) -> Value {
        Value::constant(val, Type::scalar(*self))
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

    pub const fn min_vector_size(&self) -> u8 {
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

    pub fn is_bool(&self) -> bool {
        matches!(self, ElemType::Bool)
    }

    pub fn as_float(&self) -> Option<FloatKind> {
        match self {
            ElemType::Float(kind) => Some(*kind),
            _ => None,
        }
    }

    pub fn max_variable(&self) -> Value {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MAX,
                FloatKind::E2M3 => e2m3::MAX,
                FloatKind::E3M2 => e3m2::MAX,
                FloatKind::E4M3 => e4m3::MAX.to_f64(),
                FloatKind::E5M2 => e5m2::MAX.to_f64(),
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

        Value {
            kind: ValueKind::Constant(value),
            ty: Type::scalar(*self),
        }
    }

    pub fn min_variable(&self) -> Value {
        let value = match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MIN,
                FloatKind::E2M3 => e2m3::MIN,
                FloatKind::E3M2 => e3m2::MIN,
                FloatKind::E4M3 => e4m3::MIN.to_f64(),
                FloatKind::E5M2 => e5m2::MIN.to_f64(),
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

        Value {
            kind: ValueKind::Constant(value),
            ty: Type::scalar(*self),
        }
    }

    pub fn epsilon(&self) -> f64 {
        match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => 0.5 * (e2m1::MAX - e2m1::MIN),
                FloatKind::E2M3 => 0.5 * (e2m3::MAX - e2m3::MIN),
                FloatKind::E3M2 => 0.5 * (e3m2::MAX - e3m2::MIN),
                FloatKind::E4M3 => 0.5 * (e4m3::MAX.to_f64() - e4m3::MIN.to_f64()),
                FloatKind::E5M2 => 0.5 * (e5m2::MAX.to_f64() - e5m2::MIN.to_f64()),
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
            OpaqueType::BarrierToken(_) => 8,
            OpaqueType::TensorMap => 128,
        }
    }

    /// Get the size in bits.
    pub const fn size_bits(&self) -> usize {
        self.size() * 8
    }
}

impl StorageType {
    pub fn elem_type(&self) -> ElemType {
        match self {
            StorageType::Scalar(ty) | StorageType::Packed(ty, _) => *ty,
        }
    }

    pub fn packing_factor(&self) -> usize {
        match self {
            StorageType::Packed(_, factor) => *factor,
            _ => 1,
        }
    }

    pub fn size(&self) -> usize {
        self.size_bits().div_ceil(8)
    }

    pub fn size_bits(&self) -> usize {
        match self {
            StorageType::Packed(ty, factor) => ty.size_bits() * *factor,
            StorageType::Scalar(ty) => ty.size_bits(),
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

    pub fn is_bool(&self) -> bool {
        self.elem_type().is_bool()
    }

    /// Returns an empirical epsilon for this storage type, taking quantization into account.
    pub fn epsilon(&self) -> f64 {
        match self {
            StorageType::Scalar(ty) => ty.epsilon(),
            StorageType::Packed(ty, factor) => {
                // For packed types, we can conservatively scale epsilon by the number of packed elements
                ty.epsilon() * (*factor as f64)
            }
        }
    }

    pub fn constant(&self, value: ConstantValue) -> Value {
        Value::constant(value, Type::new(*self))
    }
}

macro_rules! storage_from_elem {
    ($($ty: ty),*) => {
        $(impl From<$ty> for StorageType {
            fn from(value: $ty) -> Self {
                StorageType::Scalar(value.into())
            }
        })*
    };
}

storage_from_elem!(FloatKind, IntKind, UIntKind, ElemType);

impl From<OpaqueType> for Type {
    fn from(val: OpaqueType) -> Self {
        Type::Opaque(val)
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

/// Class of a pointer. For `Global`, the ID contains the underlying buffer ID.
/// The ID can be used to determine more detailed buffer properties, i.e. for Metal where readability
/// is part of the pointer class.
/// For ``CubeCL`` semantics, pointers classes to different buffer IDs should be treated as entirely
/// separate types.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AddressSpace {
    Global(Id),
    Shared,
    Local,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Scalar type containing a single storage element
    Scalar(StorageType),
    /// Opaque types that can be stored but not interacted with normally. i.e. barrier,
    /// arrival tokens and tensor map descriptor.
    Opaque(OpaqueType),
    /// Vector wrapping `n` storage elements
    Vector(Intern<Type>, VectorSize),
    /// No defined physical representation, purely semantic. i.e. barrier, pipeline
    Semantic(SemanticType),
    /// Atomically accessed version of `Type`
    Atomic(Intern<Type>),
    /// Pointer of `Type` into a `PointerClass`
    Pointer(Intern<Type>, AddressSpace),
    /// Statically sized array of `Type`s
    Array(Intern<Type>, usize),
    /// Dynamically sized array of `Type`s
    DynamicArray(Intern<Type>),
    /// Cooperative Matrix
    Matrix(MatrixType),
    Aggregate(AggregateKind),
}

/// `Intern` hashes the pointer, not the values, leading to unstable hashes across runs.
/// Fix this by manually hashing the value.
impl core::hash::Hash for Type {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Type::Scalar(storage_type) => storage_type.hash(state),
            Type::Opaque(opaque) => opaque.hash(state),
            Type::Vector(intern, _) => intern.as_ref().hash(state),
            Type::Semantic(semantic_type) => semantic_type.hash(state),
            Type::Atomic(intern) => intern.as_ref().hash(state),
            Type::Pointer(intern, addr_space) => {
                intern.as_ref().hash(state);
                addr_space.hash(state);
            }
            Type::Array(intern, size) => {
                intern.as_ref().hash(state);
                size.hash(state);
            }
            Type::DynamicArray(intern) => {
                intern.as_ref().hash(state);
            }
            Type::Matrix(matrix_type) => {
                matrix_type.hash(state);
            }
            Type::Aggregate(aggregate_kind) => {
                aggregate_kind.hash(state);
            }
        }
    }
}

pub type VectorSize = usize;

impl Type {
    pub fn intern(self) -> Intern<Type> {
        Intern::new(self)
    }

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

    pub fn atomic(ty: impl Into<Type>) -> Self {
        Self::Atomic(ty.into().intern())
    }

    pub fn with_vector_size(self, vector_size: VectorSize) -> Self {
        match self {
            Type::Scalar(inner) if vector_size > 1 => {
                Type::Vector(Type::new(inner).intern(), vector_size)
            }
            Type::Opaque(opaque) => Type::Opaque(opaque),
            Type::Vector(inner, _) if vector_size <= 1 => *inner,
            Type::Vector(inner, _) => Type::Vector(inner, vector_size),
            Type::Atomic(inner) => Type::Atomic(inner.with_vector_size(vector_size).intern()),
            Type::Pointer(inner, class) => {
                Type::Pointer(inner.with_vector_size(vector_size).intern(), class)
            }
            Type::Array(inner, size) => {
                Type::Array(inner.with_vector_size(vector_size).intern(), size)
            }
            Type::DynamicArray(inner) => {
                Type::DynamicArray(inner.with_vector_size(vector_size).intern())
            }
            Type::Aggregate(AggregateKind::Ptr { inner_ty, meta }) => {
                Type::Aggregate(AggregateKind::Ptr {
                    inner_ty: inner_ty.with_vector_size(vector_size).intern(),
                    meta,
                })
            }
            this @ (Type::Scalar(_) | Type::Semantic(_) | Type::Matrix(_)) => this,
        }
    }

    pub fn pointer(ty: impl Into<Type>, class: AddressSpace) -> Self {
        Self::Pointer(ty.into().intern(), class)
    }

    pub fn array(ty: impl Into<Type>, size: usize) -> Self {
        Self::Array(ty.into().intern(), size)
    }

    pub fn vector_size(&self) -> VectorSize {
        match self {
            Type::Scalar(_) => 1,
            Type::Opaque(_) => 1,
            Type::Vector(inner, vector_size) => inner.vector_size() * *vector_size,
            Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..)
            | Type::Atomic(inner)
            | Type::Pointer(inner, _) => inner.vector_size(),
            Type::Semantic(_) => 0,
            Type::Matrix(_) => 1,
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.vector_size(),
        }
    }

    pub fn array_size(&self) -> usize {
        match self {
            Type::Array(_, size) => *size,
            Type::Scalar(_) => 1,
            Type::Opaque(_) => 1,
            Type::Vector(inner, _) | Type::Atomic(inner) | Type::Pointer(inner, _) => {
                inner.array_size()
            }
            Type::Semantic(_) | Type::DynamicArray(..) => 0,
            Type::Matrix(_) => 1,
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.array_size(),
        }
    }

    pub fn align(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size(),
            Type::Opaque(opaque) => opaque.size(),
            Type::Vector(ty, vector_size) => ty.size() * *vector_size,
            Type::Atomic(inner) => inner.align(),
            Type::Array(inner, _) => inner.align(),
            Type::DynamicArray(inner, ..) => inner.align(),
            // All platforms use at least conceptually 64-bit pointers
            Type::Pointer(..) => align_of::<u64>(),
            Type::Semantic(_) => 0,
            Type::Matrix(mat) => mat.storage.size(),
            Type::Aggregate(..) => panic!("Can't get size of opaque type `Aggregate`"),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size(),
            Type::Opaque(opaque) => opaque.size(),
            Type::Vector(ty, vector_size) => ty.size() * *vector_size,
            Type::Atomic(inner) => inner.size(),
            Type::Array(inner, size) => inner.size() * *size,
            Type::DynamicArray(inner, ..) => inner.size(),
            // All platforms use at least conceptually 64-bit pointers
            Type::Pointer(..) => size_of::<u64>(),
            Type::Semantic(_) => 0,
            Type::Matrix(..) => panic!("Can't get size of opaque type `Matrix`"),
            Type::Aggregate(..) => panic!("Can't get size of opaque type `Aggregate`"),
        }
    }

    pub fn size_bits(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size_bits(),
            Type::Opaque(opaque) => opaque.size_bits(),
            Type::Vector(ty, vector_size) => ty.size_bits() * *vector_size,
            Type::Atomic(inner) => inner.size_bits(),
            Type::Array(inner, ..) => inner.size_bits(),
            Type::DynamicArray(inner, ..) => inner.size_bits(),
            // All platforms use at least conceptually 64-bit pointers
            Type::Pointer(..) => u64::BITS as usize,
            Type::Semantic(_) => 0,
            Type::Matrix(..) => panic!("Can't get size of opaque type `Matrix`"),
            Type::Aggregate(..) => panic!("Can't get size of opaque type `Aggregate`"),
        }
    }

    pub fn packing_factor(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.packing_factor(),
            Type::Opaque(_) => 1,
            Type::Vector(ty, _)
            | Type::Atomic(ty)
            | Type::Pointer(ty, _)
            | Type::Array(ty, ..)
            | Type::DynamicArray(ty, ..) => ty.packing_factor(),
            Type::Semantic(_) => 1,
            Type::Matrix(mat) => mat.storage.packing_factor(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.packing_factor(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        match self {
            Type::Semantic(_) | Type::Scalar(_) | Type::Matrix(_) | Type::Opaque(_) => false,
            Type::Atomic(_) => true,
            Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_atomic(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_atomic(),
        }
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, Type::Pointer(..))
    }

    pub fn is_int(&self) -> bool {
        match self {
            Type::Scalar(ty) => ty.is_int(),
            Type::Semantic(_) | Type::Opaque(_) => false,
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_int(),
            Type::Matrix(matrix_type) => matrix_type.storage.is_int(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_int(),
        }
    }

    pub fn is_signed_int(&self) -> bool {
        match self {
            Type::Scalar(ty) => ty.is_signed_int(),
            Type::Semantic(_) | Type::Opaque(_) => false,
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_signed_int(),
            Type::Matrix(matrix_type) => matrix_type.storage.is_signed_int(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_signed_int(),
        }
    }

    pub fn is_unsigned_int(&self) -> bool {
        match self {
            Type::Scalar(ty) => ty.is_unsigned_int(),
            Type::Semantic(_) | Type::Opaque(_) => false,
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_unsigned_int(),
            Type::Matrix(matrix_type) => matrix_type.storage.is_unsigned_int(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_unsigned_int(),
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Type::Scalar(ty) => ty.is_float(),
            Type::Semantic(_) | Type::Opaque(_) => false,
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_float(),
            Type::Matrix(matrix_type) => matrix_type.storage.is_float(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_float(),
        }
    }

    pub fn is_bool(&self) -> bool {
        match self {
            Type::Scalar(ty) => ty.is_bool(),
            Type::Semantic(_) | Type::Opaque(_) => false,
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.is_bool(),
            Type::Matrix(matrix_type) => matrix_type.storage.is_bool(),
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.is_bool(),
        }
    }

    pub fn storage_type(&self) -> StorageType {
        match self {
            Type::Scalar(ty) => *ty,
            Type::Semantic(_) | Type::Opaque(_) => {
                unimplemented!("Can't get storage for semantic type")
            }
            Type::Atomic(inner)
            | Type::Pointer(inner, _)
            | Type::Vector(inner, _)
            | Type::Array(inner, ..)
            | Type::DynamicArray(inner, ..) => inner.storage_type(),
            Type::Matrix(matrix_type) => matrix_type.storage,
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.storage_type(),
        }
    }

    pub fn as_scalar(&self) -> Self {
        match self {
            Type::Scalar(_) => *self,
            Type::Vector(inner, _) => inner.as_scalar(),
            Type::Atomic(inner) => Type::Atomic(inner.as_scalar().intern()),
            Type::Pointer(inner, class) => Type::Pointer(inner.as_scalar().intern(), *class),
            Type::Array(inner, size) => Type::Array(inner.as_scalar().intern(), *size),
            Type::Opaque(opaque_type) => Type::Opaque(*opaque_type),
            Type::Semantic(semantic_type) => Type::Semantic(*semantic_type),
            Type::DynamicArray(inner) => Type::DynamicArray(inner.as_scalar().intern()),
            Type::Matrix(matrix_type) => Type::Matrix(*matrix_type),
            Type::Aggregate(aggregate_kind) => Type::Aggregate(*aggregate_kind),
        }
    }

    /// Utility mainly for use in `cubecl-cpu`
    pub fn scalar_value_type(&self) -> Self {
        self.value_type().as_scalar()
    }

    pub fn is_semantic(&self) -> bool {
        matches!(self, Type::Semantic(_))
    }

    pub fn constant(&self, value: ConstantValue) -> Value {
        Value::constant(value, *self)
    }

    pub fn unwrap_ptr(&self) -> Type {
        match self {
            Type::Pointer(inner, _) => **inner,
            other => *other,
        }
    }

    pub fn address_space(&self) -> Option<AddressSpace> {
        match self {
            Type::Scalar(..)
            | Type::Opaque(..)
            | Type::Vector(..)
            | Type::Semantic(..)
            | Type::Atomic(..)
            | Type::Matrix(..)
            | Type::Array(..)
            | Type::DynamicArray(..)
            | Type::Aggregate(..) => None,
            Type::Pointer(.., address_space) => Some(*address_space),
        }
    }

    pub fn value_type(&self) -> Type {
        match self {
            Type::Pointer(inner, _) | Type::Array(inner, ..) | Type::DynamicArray(inner, ..) => {
                inner.value_type()
            }
            this @ (Type::Scalar(..)
            | Type::Vector(..)
            | Type::Semantic(..)
            | Type::Atomic(..)
            | Type::Matrix(..)
            | Type::Opaque(_)) => *this,
            Type::Aggregate(AggregateKind::Ptr { inner_ty, .. }) => inner_ty.value_type(),
        }
    }

    pub fn is_array_like(&self) -> bool {
        matches!(self, Type::Array(..) | Type::DynamicArray(..))
    }

    /// Whether a type is destructurable. This implies that
    /// * it does not have dynamic field offsets (i.e. `Array`)
    /// * it can exist in registers (i.e. no `Barrier` or `Atomic`)
    pub fn is_destructurable(&self) -> bool {
        match self {
            Type::Scalar(..) | Type::Vector(..) => true,
            // Should be `true`, but semantics are too dodgy right now. They're registers, but CUDA
            // wmma uses pointers for all matrix ops. So we need to keep them in memory for now.
            Type::Matrix(..) => false,
            Type::Pointer(..)
            | Type::Array(..)
            | Type::DynamicArray(..)
            | Type::Semantic(..)
            | Type::Atomic(..)
            | Type::Aggregate(..) => false,
            Type::Opaque(opaque) => match opaque {
                // Can only exist in memory
                OpaqueType::Barrier(..) | OpaqueType::TensorMap => false,
                OpaqueType::BarrierToken(..) => true,
            },
        }
    }

    pub fn is_value(&self) -> bool {
        self.value_type() == *self
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Type::Semantic(ty) => write!(f, "{ty}"),
            Type::Opaque(ty) => write!(f, "{ty}"),
            Type::Scalar(ty) => write!(f, "{ty}"),
            Type::Vector(ty, vector_size) => write!(f, "vector<{ty}, {vector_size}>"),
            Type::Atomic(ty) => write!(f, "atomic<{ty}>"),
            Type::Pointer(ty, addr_space) => write!(f, "ptr<{ty}, {addr_space}>"),
            Type::Array(ty, size) => write!(f, "array<{ty}, {size}>"),
            Type::DynamicArray(ty) => write!(f, "array<{ty}>"),
            Type::Matrix(mat) => write!(
                f,
                "matrix<{}, m{}xn{}xk{}x{}, {}, {}>",
                mat.ident, mat.m, mat.n, mat.k, mat.storage, mat.layout, mat.storage
            ),
            Type::Aggregate(aggregate_kind) => write!(f, "{aggregate_kind}"),
        }
    }
}

impl Display for StorageType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StorageType::Scalar(ty) => write!(f, "{ty}"),
            StorageType::Packed(ty, factor) => write!(f, "packed<{ty}, {factor}>"),
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
            SemanticType::TensorLayout(dims, _) => write!(f, "tensor_layout<{dims}>"),
            SemanticType::TensorView(dims, has_dims, permutation) => {
                write!(
                    f,
                    "tensor_layout<{:?}, has_dims: {has_dims}>",
                    &permutation[..*dims]
                )
            }
        }
    }
}

impl Display for OpaqueType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OpaqueType::Barrier(level) => write!(f, "barrier<{level}>"),
            OpaqueType::BarrierToken(level) => write!(f, "barrier_token<{level}>"),
            OpaqueType::TensorMap => f.write_str("tensor_map"),
        }
    }
}

impl Display for AddressSpace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AddressSpace::Global(id) => write!(f, "global<{id}>"),
            AddressSpace::Shared => write!(f, "shared"),
            AddressSpace::Local => f.write_str("local"),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord, Display)]
pub enum AggregateKind {
    #[display("ptr<{meta}, {inner_ty}>")]
    Ptr {
        inner_ty: Intern<Type>,
        meta: MetadataKind,
    },
}

impl AggregateKind {
    pub fn ptr(inner_ty: Type, meta: MetadataKind) -> Self {
        AggregateKind::Ptr {
            inner_ty: inner_ty.intern(),
            meta,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord, Display)]
pub enum MetadataKind {
    /// Slice metadata (offset and length)
    #[display("slice")]
    Slice,
    /// Bounds check (in bounds)
    #[display("bounds_checked")]
    BoundsCheck,
}

pub struct BoundsCheckMetadata;
impl BoundsCheckMetadata {
    pub const POINTER: usize = 0;
    pub const IS_IN_BOUNDS: usize = 1;
}

pub struct SliceMetadata;
impl SliceMetadata {
    pub const LIST: usize = 0;
    pub const OFFSET: usize = 1;
    pub const LENGTH: usize = 2;
}

impl From<e2m1x2> for Value {
    fn from(_value: e2m1x2) -> Self {
        unimplemented!("Can't currently construct e2m1x2")
    }
}

impl From<e2m3> for Value {
    fn from(_value: e2m3) -> Self {
        unimplemented!("Can't currently construct fp6")
    }
}

impl From<e3m2> for Value {
    fn from(_value: e3m2) -> Self {
        unimplemented!("Can't currently construct fp6")
    }
}

impl From<i8> for ConstantValue {
    fn from(value: i8) -> Self {
        ConstantValue::Int(value as i64)
    }
}

impl From<i16> for ConstantValue {
    fn from(value: i16) -> Self {
        ConstantValue::Int(value as i64)
    }
}

impl From<i32> for ConstantValue {
    fn from(value: i32) -> Self {
        ConstantValue::Int(value as i64)
    }
}

impl From<isize> for ConstantValue {
    fn from(value: isize) -> Self {
        ConstantValue::Int(value as i64)
    }
}

impl From<u8> for ConstantValue {
    fn from(value: u8) -> Self {
        ConstantValue::UInt(value as u64)
    }
}

impl From<u16> for ConstantValue {
    fn from(value: u16) -> Self {
        ConstantValue::UInt(value as u64)
    }
}

impl From<u32> for ConstantValue {
    fn from(value: u32) -> Self {
        ConstantValue::UInt(value as u64)
    }
}

impl From<usize> for ConstantValue {
    fn from(value: usize) -> Self {
        ConstantValue::UInt(value as u64)
    }
}

impl From<e2m1> for ConstantValue {
    fn from(value: e2m1) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<e4m3> for ConstantValue {
    fn from(value: e4m3) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<e5m2> for ConstantValue {
    fn from(value: e5m2) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<ue8m0> for ConstantValue {
    fn from(value: ue8m0) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<half::f16> for ConstantValue {
    fn from(value: half::f16) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<half::bf16> for ConstantValue {
    fn from(value: half::bf16) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<flex32> for ConstantValue {
    fn from(value: flex32) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<tf32> for ConstantValue {
    fn from(value: tf32) -> Self {
        ConstantValue::Float(value.to_f64())
    }
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        ConstantValue::Float(value as f64)
    }
}

macro_rules! impl_into_value {
    ($($ty: ty => $kind: path,)*) => {
        $(
            impl From<$ty> for Value {
                fn from(value: $ty) -> Self {
                    Value {kind: ValueKind::Constant(value.into()), ty: $kind.into()}
                }
            }
        )*
    };
}

impl_into_value!(
    bool => ElemType::Bool,

    i8 => IntKind::I8,
    i16 => IntKind::I16,
    i32 => IntKind::I32,
    i64 => IntKind::I64,

    u8 => UIntKind::U8,
    u16 => UIntKind::U16,
    u32 => UIntKind::U32,
    u64 => UIntKind::U64,

    e2m1 => FloatKind::E2M1,
    e4m3 => FloatKind::E4M3,
    e5m2 => FloatKind::E5M2,
    ue8m0 => FloatKind::UE8M0,
    f16 => FloatKind::F16,
    bf16 => FloatKind::BF16,
    f32 => FloatKind::F32,
    flex32 => FloatKind::Flex32,
    tf32 => FloatKind::TF32,
    f64 => FloatKind::F64,

    usize => UIntKind::U32,
    isize => IntKind::I32,
);

#[cfg(test)]
mod enum_counts_tests {
    use super::*;

    #[test]
    fn count_and_index_are_consistent() {
        assert_eq!(FloatKind::COUNT, 12);
        assert_eq!(FloatKind::VARIANTS.len(), FloatKind::COUNT);
        for (i, kind) in FloatKind::VARIANTS.into_iter().enumerate() {
            assert_eq!(kind.index(), i);
        }
    }

    #[test]
    fn named_fields_match_indexed_access() {
        let mut counts = FloatKindCounts::<u32>::default();
        counts.f16 = 7;
        counts[FloatKind::F32] = 9;

        assert_eq!(counts[FloatKind::F16], 7);
        assert_eq!(counts.f32, 9);
        // as_slice is in declaration order, so index() addresses the same entry.
        assert_eq!(counts.as_slice()[FloatKind::F16.index()], 7);
        assert_eq!(counts.iter().filter(|(_, v)| **v != 0).count(), 2);
    }

    #[test]
    fn is_pod() {
        // Round-trips through bytes: relies on the generated `#[repr(C)]` + `Pod` impl.
        let mut counts = UIntKindCounts::<u32>::default();
        counts.u32 = 42;
        let bytes = bytemuck::bytes_of(&counts);
        assert_eq!(bytes.len(), UIntKind::COUNT * size_of::<u32>());
        let back: UIntKindCounts<u32> = *bytemuck::from_bytes(bytes);
        assert_eq!(back.u32, 42);
    }
}
