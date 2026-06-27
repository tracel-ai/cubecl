use super::{ConstantValue, ExpandValue};
use crate::{
    AddressType, ContextExt, Scope, TypeHash,
    types::{scalar::*, spirv::ClampMode},
};
use core::fmt::Display;
use cubecl_common::{
    e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32,
    quant::scheme::{QuantParam, QuantValue},
    tf32, ue8m0,
};
use derive_more::{Display, From};
use half::{bf16, f16};

pub use internment::Intern;
use pliron::{context::Context, derive::format, r#type::TypeHandle};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum FloatKind {
    /// FP4, 2 bit exponent, 1 bit mantissa
    E2M1,
    /// `FP4x2`, 2 bit exponent, 1 bit mantissa
    E2M1x2,
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

impl FloatKind {
    pub fn to_type(&self, ctx: &Context) -> TypeHandle {
        match self {
            FloatKind::E2M1 => Float4E2M1Type::get(ctx).into(),
            FloatKind::E2M1x2 => Float4E2M1x2Type::get(ctx).into(),
            FloatKind::E2M3 => Float6E2M3Type::get(ctx).into(),
            FloatKind::E3M2 => Float6E3M2Type::get(ctx).into(),
            FloatKind::E4M3 => Float8E4M3Type::get(ctx).into(),
            FloatKind::E5M2 => Float8E5M2Type::get(ctx).into(),
            FloatKind::UE8M0 => Float8E8M0Type::get(ctx).into(),
            FloatKind::F16 => Float16Type::get(ctx).into(),
            FloatKind::BF16 => BFloat16Type::get(ctx).into(),
            FloatKind::Flex32 => FloatFlex32Type::get(ctx).into(),
            FloatKind::F32 => Float32Type::get(ctx).into(),
            FloatKind::TF32 => TFloat32Type::get(ctx).into(),
            FloatKind::F64 => Float64Type::get(ctx).into(),
        }
    }
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

impl IntKind {
    pub fn to_type(&self, ctx: &Context) -> TypeHandle {
        IntType::get(ctx, self.size_bits()).into()
    }

    pub fn size_bits(&self) -> usize {
        match self {
            IntKind::I8 => 8,
            IntKind::I16 => 16,
            IntKind::I32 => 32,
            IntKind::I64 => 64,
        }
    }
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

impl UIntKind {
    pub fn to_type(&self, ctx: &Context) -> TypeHandle {
        UIntType::get(ctx, self.size_bits()).into()
    }

    pub fn size_bits(&self) -> usize {
        match self {
            UIntKind::U8 => 8,
            UIntKind::U16 => 16,
            UIntKind::U32 => 32,
            UIntKind::U64 => 64,
        }
    }
}

/// Conceptual element type, not necessarily the physical type used in the code
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, From)]
#[allow(missing_docs)]
pub enum ElemType {
    Index,
    Float(FloatKind),
    Int(IntKind),
    UInt(UIntKind),
    Bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpaqueType {
    Barrier,
    TensorMap,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemanticType {
    TensorLayout(usize, ClampMode),
    TensorView(usize, bool, [u32; 5]),
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

    pub fn to_type(&self, ctx: &Context) -> TypeHandle {
        match self {
            ElemType::Index => IndexType::get(ctx).into(),
            ElemType::Float(float_kind) => float_kind.to_type(ctx),
            ElemType::Int(int_kind) => int_kind.to_type(ctx),
            ElemType::UInt(uint_kind) => uint_kind.to_type(ctx),
            ElemType::Bool => BoolType::get(ctx).into(),
        }
    }

    /// Create a constant from a constant value.
    ///
    /// The output will have the same type as the element.
    pub fn constant(&self, val: ConstantValue) -> ExpandValue {
        ExpandValue::constant(val, *self)
    }

    pub fn with_vector_size(self, vector_size: VectorSize) -> Type {
        let ty = Type::Scalar(self);
        if vector_size > 1 {
            Type::Vector(ty.intern(), vector_size)
        } else {
            ty
        }
    }

    pub fn expand_size(&self, address_type: AddressType) -> usize {
        match self {
            ElemType::Index => address_type.size(),
            other => other.size(),
        }
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        match self {
            ElemType::Index => panic!("Can't get index size outside kernel"),
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1
                | FloatKind::E2M1x2
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
    pub fn size_bits(&self) -> usize {
        match self {
            ElemType::Index => panic!("Can't get index size outside kernel"),
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1x2
                | FloatKind::E2M3
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

    pub fn max_variable(&self, scope: &Scope) -> ExpandValue {
        let value = match self {
            ElemType::Index => {
                let addr = scope.ctx().address_type().unsigned_type();
                return addr.max_variable(scope);
            }
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MAX,
                FloatKind::E2M1x2 => e2m1::MAX,
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

        ExpandValue::Constant { value, ty: *self }
    }

    pub fn min_variable(&self) -> ExpandValue {
        let value = match self {
            ElemType::Index => 0u64.into(),
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::MIN,
                FloatKind::E2M1x2 => e2m1::MIN,
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

        ExpandValue::Constant { value, ty: *self }
    }

    pub fn epsilon(&self) -> f64 {
        match self {
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => 0.5 * (e2m1::MAX - e2m1::MIN),
                FloatKind::E2M1x2 => 0.5 * (e2m1::MAX - e2m1::MIN),
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
            ElemType::Index | ElemType::Int(_) | ElemType::UInt(_) => 1.0, // step of 1
            ElemType::Bool => 1.0,
        }
    }
}

impl From<OpaqueType> for Type {
    fn from(val: OpaqueType) -> Self {
        Type::Opaque(val)
    }
}

impl<T: Into<ElemType>> From<T> for Type {
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
#[format]
pub enum AddressSpace {
    #[format("`<` $0 `>`")]
    Global(usize),
    Shared,
    Local,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Scalar type containing a single storage element
    Scalar(ElemType),
    /// Opaque types that can be stored but not interacted with normally. i.e. barrier,
    /// arrival tokens and tensor map descriptor.
    Opaque(OpaqueType),
    /// Vector wrapping `n` storage elements
    Vector(Intern<Type>, VectorSize),
    /// No defined physical representation, purely semantic. i.e. barrier, pipeline
    Semantic(SemanticType),
    /// Atomically accessed version of `Type`
    Atomic(Intern<Type>),
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
        }
    }
}

pub type VectorSize = usize;

impl Type {
    pub fn intern(self) -> Intern<Type> {
        Intern::new(self)
    }

    /// Create a new type
    pub fn new(elem: impl Into<ElemType>) -> Self {
        Type::Scalar(elem.into())
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
            this @ (Type::Scalar(_) | Type::Semantic(_)) => this,
        }
    }

    pub fn vector_size(&self) -> VectorSize {
        match self {
            Type::Scalar(_) => 1,
            Type::Opaque(_) => 1,
            Type::Vector(inner, vector_size) => inner.vector_size() * *vector_size,
            Type::Atomic(inner) => inner.vector_size(),
            Type::Semantic(_) => 0,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Type::Scalar(ty) => ty.size(),
            Type::Opaque(_) => panic!("Can't get size of opaque type"),
            Type::Vector(ty, vector_size) => ty.size() * *vector_size,
            Type::Atomic(inner) => inner.size(),
            Type::Semantic(_) => 0,
        }
    }

    pub fn elem_type(&self) -> ElemType {
        match self {
            Type::Scalar(ty) => *ty,
            Type::Semantic(_) | Type::Opaque(_) => {
                unimplemented!("Can't get storage for semantic type")
            }
            Type::Atomic(inner) | Type::Vector(inner, _) => inner.elem_type(),
        }
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
        }
    }
}

impl Display for ElemType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Index => f.write_str("usize"),
            Self::Float(kind) => match kind {
                FloatKind::E2M1 => f.write_str("e2m1"),
                FloatKind::E2M1x2 => f.write_str("e2m1x2"),
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
            OpaqueType::Barrier => write!(f, "barrier"),
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

impl From<e2m1x2> for ExpandValue {
    fn from(_value: e2m1x2) -> Self {
        unimplemented!("Can't currently construct e2m1x2")
    }
}

impl From<e2m3> for ExpandValue {
    fn from(_value: e2m3) -> Self {
        unimplemented!("Can't currently construct fp6")
    }
}

impl From<e3m2> for ExpandValue {
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
            impl From<$ty> for ExpandValue {
                fn from(value: $ty) -> Self {
                    ExpandValue::Constant { value: value.into(), ty: $kind.into() }
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

    usize => ElemType::Index,
    isize => IntKind::I32,
);
