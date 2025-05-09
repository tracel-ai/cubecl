use super::{ConstantScalarValue, Variable, VariableKind};
use crate::TypeHash;
use core::fmt::Display;
use core::num::NonZero;
use cubecl_common::{e2m1, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};

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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum Elem {
    Float(FloatKind),
    Int(IntKind),
    UInt(UIntKind),
    AtomicFloat(FloatKind),
    AtomicInt(IntKind),
    AtomicUInt(UIntKind),
    Bool,
}

impl Elem {
    /// Create a constant scalar from a float.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_f64(&self, val: f64) -> Variable {
        Variable::constant(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::Bool => ConstantScalarValue::Bool(val > 0.0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicUInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::AtomicFloat(kind) => ConstantScalarValue::Float(val, *kind),
        })
    }
    /// Create a constant scalar from a signed integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_i64(&self, val: i64) -> Variable {
        Variable::constant(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val, *kind),
            Elem::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::Bool => ConstantScalarValue::Bool(val > 0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val, *kind),
            Elem::AtomicUInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::AtomicFloat(kind) => ConstantScalarValue::Float(val as f64, *kind),
        })
    }
    /// Create a constant scalar from a unsigned integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_u64(&self, val: u64) -> Variable {
        Variable::constant(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt(kind) => ConstantScalarValue::UInt(val, *kind),
            Elem::Bool => ConstantScalarValue::Bool(val > 0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicUInt(kind) => ConstantScalarValue::UInt(val, *kind),
            Elem::AtomicFloat(kind) => ConstantScalarValue::Float(val as f64, *kind),
        })
    }
    /// Create a constant scalar from a boolean.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_bool(&self, val: bool) -> Variable {
        Variable::constant(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as u32 as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::AtomicUInt(kind) => ConstantScalarValue::UInt(val as u64, *kind),
            Elem::AtomicFloat(kind) => ConstantScalarValue::Float(val as u32 as f64, *kind),
            Elem::Bool => ConstantScalarValue::Bool(val),
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
            Elem::Float(kind) | Elem::AtomicFloat(kind) => match kind {
                FloatKind::E2M1 => panic!("Can't get byte size of sub-byte type"),
                FloatKind::E2M3
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
            Elem::Int(kind) | Elem::AtomicInt(kind) => match kind {
                IntKind::I8 => core::mem::size_of::<i8>(),
                IntKind::I16 => core::mem::size_of::<i16>(),
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            Elem::UInt(kind) | Elem::AtomicUInt(kind) => match kind {
                UIntKind::U8 => core::mem::size_of::<u8>(),
                UIntKind::U16 => core::mem::size_of::<u16>(),
                UIntKind::U32 => core::mem::size_of::<u32>(),
                UIntKind::U64 => core::mem::size_of::<u64>(),
            },
            Elem::Bool => core::mem::size_of::<bool>(),
        }
    }

    /// Get the size in bytes.
    pub const fn size_bits(&self) -> usize {
        match self {
            Elem::Float(kind) | Elem::AtomicFloat(kind) => match kind {
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
            Elem::Int(_)
            | Elem::AtomicInt(_)
            | Elem::UInt(_)
            | Elem::AtomicUInt(_)
            | Elem::Bool => self.size() * 8,
        }
    }

    pub const fn min_line_size(&self) -> u8 {
        match self {
            Elem::Float(FloatKind::E2M1) => 2,
            _ => 1,
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(
            self,
            Elem::AtomicFloat(_) | Elem::AtomicInt(_) | Elem::AtomicUInt(_)
        )
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            Elem::Int(_) | Elem::AtomicInt(_) | Elem::UInt(_) | Elem::AtomicUInt(_)
        )
    }

    pub fn max_variable(&self) -> Variable {
        let value = match self {
            Elem::Float(kind) | Elem::AtomicFloat(kind) => match kind {
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
            Elem::Int(kind) | Elem::AtomicInt(kind) => match kind {
                IntKind::I8 => ConstantScalarValue::Int(i8::MAX.into(), IntKind::I8),
                IntKind::I16 => ConstantScalarValue::Int(i16::MAX.into(), IntKind::I16),
                IntKind::I32 => ConstantScalarValue::Int(i32::MAX.into(), IntKind::I32),
                IntKind::I64 => ConstantScalarValue::Int(i64::MAX, IntKind::I64),
            },
            Elem::UInt(kind) | Elem::AtomicUInt(kind) => match kind {
                UIntKind::U8 => ConstantScalarValue::UInt(u8::MAX.into(), UIntKind::U8),
                UIntKind::U16 => ConstantScalarValue::UInt(u16::MAX.into(), UIntKind::U16),
                UIntKind::U32 => ConstantScalarValue::UInt(u32::MAX.into(), UIntKind::U32),
                UIntKind::U64 => ConstantScalarValue::UInt(u64::MAX, UIntKind::U64),
            },
            Elem::Bool => ConstantScalarValue::Bool(true),
        };

        Variable::new(VariableKind::ConstantScalar(value), Item::new(*self))
    }

    pub fn min_variable(&self) -> Variable {
        let value = match self {
            Elem::Float(kind) | Elem::AtomicFloat(kind) => match kind {
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
            Elem::Int(kind) | Elem::AtomicInt(kind) => match kind {
                IntKind::I8 => ConstantScalarValue::Int(i8::MIN.into(), IntKind::I8),
                IntKind::I16 => ConstantScalarValue::Int(i16::MIN.into(), IntKind::I16),
                IntKind::I32 => ConstantScalarValue::Int(i32::MIN.into(), IntKind::I32),
                IntKind::I64 => ConstantScalarValue::Int(i64::MIN, IntKind::I64),
            },
            Elem::UInt(kind) | Elem::AtomicUInt(kind) => match kind {
                UIntKind::U8 => ConstantScalarValue::UInt(u8::MIN.into(), UIntKind::U8),
                UIntKind::U16 => ConstantScalarValue::UInt(u16::MIN.into(), UIntKind::U16),
                UIntKind::U32 => ConstantScalarValue::UInt(u32::MIN.into(), UIntKind::U32),
                UIntKind::U64 => ConstantScalarValue::UInt(u64::MIN, UIntKind::U64),
            },
            Elem::Bool => ConstantScalarValue::Bool(false),
        };

        Variable::new(VariableKind::ConstantScalar(value), Item::new(*self))
    }
}

impl From<Elem> for Item {
    fn from(val: Elem) -> Self {
        Item::new(val)
    }
}

impl Display for Elem {
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
            Self::AtomicFloat(kind) => write!(f, "atomic<{}>", Elem::Float(*kind)),
            Self::Int(kind) => match kind {
                IntKind::I8 => f.write_str("i8"),
                IntKind::I16 => f.write_str("i16"),
                IntKind::I32 => f.write_str("i32"),
                IntKind::I64 => f.write_str("i64"),
            },
            Self::AtomicInt(kind) => write!(f, "atomic<{}>", Elem::Int(*kind)),
            Self::UInt(kind) => match kind {
                UIntKind::U8 => f.write_str("u8"),
                UIntKind::U16 => f.write_str("u16"),
                UIntKind::U32 => f.write_str("u32"),
                UIntKind::U64 => f.write_str("u64"),
            },
            Self::AtomicUInt(kind) => write!(f, "atomic<{}>", Elem::UInt(*kind)),
            Self::Bool => f.write_str("bool"),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Item {
    pub elem: Elem,
    pub vectorization: Vectorization,
}

pub type Vectorization = Option<NonZero<u8>>;

impl Item {
    /// Fetch the elem of the item.
    pub fn elem(&self) -> Elem {
        self.elem
    }

    /// Create a new item without vectorization
    pub fn new(elem: Elem) -> Self {
        Self {
            elem,
            vectorization: None,
        }
    }

    /// Create a new item with vectorization
    pub fn vectorized(elem: Elem, vectorization: Vectorization) -> Self {
        Self {
            elem,
            vectorization,
        }
    }

    pub fn vectorize(&self, vectorization: Vectorization) -> Item {
        Item {
            elem: self.elem,
            vectorization,
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.vectorization {
            Some(vec) if vec.get() > 1 => {
                write!(f, "vector{}<{}>", vec.get(), self.elem)
            }
            _ => write!(f, "{}", self.elem),
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
