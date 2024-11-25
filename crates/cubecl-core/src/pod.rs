use bytemuck::NoUninit;

use crate::{
    flex32,
    ir::{Elem, FloatKind, IntKind, UIntKind},
    prelude::Numeric,
};

/// The base element trait for the jit backend.
pub trait CubeElement: core::fmt::Debug + Send + Sync + 'static + Clone + NoUninit {
    /// Returns the name of the type.
    fn type_name() -> &'static str;
    /// Convert a slice of elements to a slice of bytes.
    fn to_elem_data(slice: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice(slice).to_vec()
    }
    /// Convert a slice of bytes to a slice of elements.
    fn from_elem_data(element_data: Vec<u8>) -> Vec<Self>;
    /// Element representation for `cubecl`.
    fn cube_elem() -> Elem;
    /// Highest possible value
    fn maximum_value() -> Self;
    /// Lowest possible value
    fn minimum_value() -> Self;
}

impl CubeElement for u64 {
    fn type_name() -> &'static str {
        "u64"
    }
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::UInt(UIntKind::U64)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::UInt(UIntKind::U32)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::UInt(UIntKind::U16)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::UInt(UIntKind::U8)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Int(IntKind::I64)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Int(IntKind::I32)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Int(IntKind::I16)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Int(IntKind::I8)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::F64)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::F32)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::F16)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::BF16)
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
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytemuck::cast_slice(bytes.as_slice()).to_vec()
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::Flex32)
    }
    fn maximum_value() -> Self {
        flex32::MAX
    }
    fn minimum_value() -> Self {
        flex32::MIN
    }
}

impl CubeElement for bool {
    fn type_name() -> &'static str {
        "bool"
    }
    fn to_elem_data(slice: &[Self]) -> Vec<u8> {
        slice
            .iter()
            .flat_map(|&x| {
                if x {
                    1u32.to_le_bytes()
                } else {
                    0u32.to_le_bytes()
                }
            })
            .collect()
    }
    fn from_elem_data(bytes: Vec<u8>) -> Vec<Self> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) != 0)
            .collect()
    }
    fn cube_elem() -> Elem {
        Elem::Bool
    }
    fn maximum_value() -> Self {
        true
    }
    fn minimum_value() -> Self {
        false
    }
}
