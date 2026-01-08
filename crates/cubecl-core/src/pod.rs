use cubecl_common::{e2m1, e2m1x2, e4m3, e5m2, flex32, tf32, ue8m0};
use cubecl_ir::StorageType;

use crate::{
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    prelude::{CubePrimitive, Numeric},
};

/// The base element trait for the jit backend.
pub trait CubeElement: core::fmt::Debug + Send + Sync + 'static + Clone + bytemuck::Pod {
    /// Returns the name of the type.
    fn type_name() -> &'static str;
    /// Convert a slice of elements to a slice of bytes.
    fn as_bytes(slice: &[Self]) -> &[u8];
    /// Convert a slice of bytes to a slice of elements.
    fn from_bytes(bytes: &[u8]) -> &[Self];
    /// Element representation for `cubecl`.
    fn cube_type() -> StorageType;
    /// Highest possible value
    fn maximum_value() -> Self;
    /// Lowest possible value
    fn minimum_value() -> Self;
}

pub trait CubeScalar: CubeElement + CubePrimitive + num_traits::NumCast {}

impl<E: CubeElement + CubePrimitive + num_traits::NumCast> CubeScalar for E {}

impl CubeElement for u64 {
    fn type_name() -> &'static str {
        "u64"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::UInt(UIntKind::U64).into()
    }
    fn maximum_value() -> Self {
        u64::MAX
    }
    fn minimum_value() -> Self {
        u64::MIN
    }
}

impl CubeElement for u32 {
    fn type_name() -> &'static str {
        "u32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::UInt(UIntKind::U32).into()
    }
    fn maximum_value() -> Self {
        u32::MAX
    }
    fn minimum_value() -> Self {
        u32::MIN
    }
}

impl CubeElement for u16 {
    fn type_name() -> &'static str {
        "u16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::UInt(UIntKind::U16).into()
    }
    fn maximum_value() -> Self {
        u16::MAX
    }
    fn minimum_value() -> Self {
        u16::MIN
    }
}

impl CubeElement for u8 {
    fn type_name() -> &'static str {
        "u8"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::UInt(UIntKind::U8).into()
    }
    fn maximum_value() -> Self {
        u8::MAX
    }
    fn minimum_value() -> Self {
        u8::MIN
    }
}

impl CubeElement for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Int(IntKind::I64).into()
    }
    fn maximum_value() -> Self {
        // Seems to cause problem for some GPU
        i64::MAX - 1
    }
    fn minimum_value() -> Self {
        // Seems to cause problem for some GPU
        i64::MIN + 1
    }
}

impl CubeElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Int(IntKind::I32).into()
    }
    fn maximum_value() -> Self {
        // Seems to cause problem for some GPU
        i32::MAX - 1
    }
    fn minimum_value() -> Self {
        // Seems to cause problem for some GPU
        i32::MIN + 1
    }
}

impl CubeElement for i16 {
    fn type_name() -> &'static str {
        "i16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Int(IntKind::I16).into()
    }
    fn maximum_value() -> Self {
        // Seems to cause problem for some GPU
        i16::MAX - 1
    }
    fn minimum_value() -> Self {
        // Seems to cause problem for some GPU
        i16::MIN + 1
    }
}

impl CubeElement for i8 {
    fn type_name() -> &'static str {
        "i8"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Int(IntKind::I8).into()
    }
    fn maximum_value() -> Self {
        // Seems to cause problem for some GPU
        i8::MAX - 1
    }
    fn minimum_value() -> Self {
        // Seems to cause problem for some GPU
        i8::MIN + 1
    }
}

impl CubeElement for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::F64).into()
    }
    fn maximum_value() -> Self {
        f64::MAX
    }
    fn minimum_value() -> Self {
        f64::MIN
    }
}

impl CubeElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::F32).into()
    }
    fn maximum_value() -> Self {
        f32::MAX
    }
    fn minimum_value() -> Self {
        f32::MIN
    }
}

impl CubeElement for half::f16 {
    fn type_name() -> &'static str {
        "f16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::F16).into()
    }
    fn maximum_value() -> Self {
        half::f16::MAX
    }
    fn minimum_value() -> Self {
        half::f16::MIN
    }
}

impl CubeElement for half::bf16 {
    fn type_name() -> &'static str {
        "bf16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::BF16).into()
    }
    fn maximum_value() -> Self {
        half::bf16::MAX
    }
    fn minimum_value() -> Self {
        half::bf16::MIN
    }
}

impl CubeElement for flex32 {
    fn type_name() -> &'static str {
        "flex32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::Flex32).into()
    }
    fn maximum_value() -> Self {
        <flex32 as num_traits::Float>::max_value()
    }
    fn minimum_value() -> Self {
        <flex32 as num_traits::Float>::min_value()
    }
}

impl CubeElement for tf32 {
    fn type_name() -> &'static str {
        "tf32"
    }

    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }

    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::TF32).into()
    }

    fn maximum_value() -> Self {
        tf32::max_value()
    }

    fn minimum_value() -> Self {
        tf32::min_value()
    }
}

impl CubeElement for e4m3 {
    fn type_name() -> &'static str {
        "e4m3"
    }

    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }

    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::E4M3).into()
    }

    fn maximum_value() -> Self {
        e4m3::from_f64(e4m3::MAX)
    }

    fn minimum_value() -> Self {
        e4m3::from_f64(e4m3::MIN)
    }
}

impl CubeElement for e5m2 {
    fn type_name() -> &'static str {
        "e5m2"
    }

    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }

    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::E5M2).into()
    }

    fn maximum_value() -> Self {
        e5m2::from_f64(e5m2::MAX)
    }

    fn minimum_value() -> Self {
        e5m2::from_f64(e5m2::MIN)
    }
}

impl CubeElement for ue8m0 {
    fn type_name() -> &'static str {
        "ue8m0"
    }

    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }

    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    fn cube_type() -> StorageType {
        ElemType::Float(FloatKind::UE8M0).into()
    }

    fn maximum_value() -> Self {
        ue8m0::from_f64(ue8m0::MAX)
    }

    fn minimum_value() -> Self {
        ue8m0::from_f64(ue8m0::MIN)
    }
}

impl CubeElement for e2m1x2 {
    fn type_name() -> &'static str {
        "e2m1x2"
    }

    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }

    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    fn cube_type() -> StorageType {
        StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)
    }

    fn maximum_value() -> Self {
        let max = e2m1::MAX.to_bits() as u8;
        e2m1x2::from_bits(max << 4 | max)
    }

    fn minimum_value() -> Self {
        let min = e2m1::MIN.to_bits() as u8;
        e2m1x2::from_bits(min << 4 | min)
    }
}
