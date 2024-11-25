use super::{ConstantScalarValue, Scope, Variable, VariableKind};
use crate::PLANE_DIM_APPROX;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::num::NonZero;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub cube_dim: CubeDim,
    pub body: Scope,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Location {
    Storage,
    Cube,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum FloatKind {
    F16,
    BF16,
    Flex32,
    F32,
    TF32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum UIntKind {
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum Elem {
    Float(FloatKind),
    Int(IntKind),
    AtomicInt(IntKind),
    UInt(UIntKind),
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
            Elem::Float(kind) => match kind {
                FloatKind::F16 => core::mem::size_of::<half::f16>(),
                FloatKind::BF16 => core::mem::size_of::<half::bf16>(),
                FloatKind::F32 => core::mem::size_of::<f32>(),
                FloatKind::F64 => core::mem::size_of::<f64>(),
                FloatKind::Flex32 => core::mem::size_of::<f32>(),
                FloatKind::TF32 => core::mem::size_of::<f32>(),
            },
            Elem::Int(kind) => match kind {
                IntKind::I8 => core::mem::size_of::<i8>(),
                IntKind::I16 => core::mem::size_of::<i16>(),
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            Elem::AtomicInt(kind) => match kind {
                IntKind::I8 => core::mem::size_of::<i8>(),
                IntKind::I16 => core::mem::size_of::<i16>(),
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            Elem::UInt(kind) => match kind {
                UIntKind::U8 => core::mem::size_of::<u8>(),
                UIntKind::U16 => core::mem::size_of::<u16>(),
                UIntKind::U32 => core::mem::size_of::<u32>(),
                UIntKind::U64 => core::mem::size_of::<u64>(),
            },
            Elem::AtomicUInt(kind) => match kind {
                UIntKind::U8 => core::mem::size_of::<u8>(),
                UIntKind::U16 => core::mem::size_of::<u16>(),
                UIntKind::U32 => core::mem::size_of::<u32>(),
                UIntKind::U64 => core::mem::size_of::<u64>(),
            },
            // Currently, bools are represented as u32 in the backend.
            Elem::Bool => core::mem::size_of::<u32>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, Elem::AtomicInt(_) | Elem::AtomicUInt(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            Elem::Int(_) | Elem::AtomicInt(_) | Elem::UInt(_) | Elem::AtomicUInt(_)
        )
    }
}

impl From<Elem> for Item {
    fn from(val: Elem) -> Self {
        Item::new(val)
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(kind) => match kind {
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
            Self::AtomicInt(kind) => match kind {
                IntKind::I8 => f.write_str("atomic<i8>"),
                IntKind::I16 => f.write_str("atomic<i16>"),
                IntKind::I32 => f.write_str("atomic<i32>"),
                IntKind::I64 => f.write_str("atomic<i64>"),
            },
            Self::UInt(kind) => match kind {
                UIntKind::U8 => f.write_str("u8"),
                UIntKind::U16 => f.write_str("u16"),
                UIntKind::U32 => f.write_str("u32"),
                UIntKind::U64 => f.write_str("u64"),
            },
            Self::AtomicUInt(kind) => match kind {
                UIntKind::U8 => f.write_str("atomic<u8>"),
                UIntKind::U16 => f.write_str("atomic<u16>"),
                UIntKind::U32 => f.write_str("atomic<u32>"),
                UIntKind::U64 => f.write_str("atomic<u64>"),
            },
            Self::Bool => f.write_str("bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub struct Item {
    pub elem: Elem,
    pub vectorization: Vectorization,
}

pub type Vectorization = Option<NonZero<u8>>;

impl Item {}

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

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Item {
        Item {
            elem: self.elem,
            vectorization,
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.vectorization {
            Some(vec) if vec.get() > 1 => {
                write!(f, "vector{}<{}>", vec.get(), self.elem)
            }
            _ => write!(f, "{}", self.elem),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
    pub has_extended_meta: bool,
}

#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct CubeDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CubeDim {
    pub fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for CubeDim {
    fn default() -> Self {
        Self {
            x: PLANE_DIM_APPROX as u32,
            y: PLANE_DIM_APPROX as u32,
            z: 1,
        }
    }
}
