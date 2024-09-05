use super::{ConstantScalarValue, Scope, Variable, Vectorization};
use crate::SUBCUBE_DIM_APPROX;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

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
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum IntKind {
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum Elem {
    Float(FloatKind),
    Int(IntKind),
    AtomicInt(IntKind),
    UInt,
    AtomicUInt,
    Bool,
}

impl Elem {
    /// Create a constant scalar from a float.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_f64(&self, val: f64) -> Variable {
        Variable::ConstantScalar(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt => ConstantScalarValue::UInt(val as u64),
            Elem::Bool => ConstantScalarValue::Bool(val > 0.0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicUInt => ConstantScalarValue::UInt(val as u64),
        })
    }
    /// Create a constant scalar from a signed integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_i64(&self, val: i64) -> Variable {
        Variable::ConstantScalar(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val, *kind),
            Elem::UInt => ConstantScalarValue::UInt(val as u64),
            Elem::Bool => ConstantScalarValue::Bool(val > 0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val, *kind),
            Elem::AtomicUInt => ConstantScalarValue::UInt(val as u64),
        })
    }
    /// Create a constant scalar from a unsigned integer.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_u64(&self, val: u64) -> Variable {
        Variable::ConstantScalar(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt => ConstantScalarValue::UInt(val),
            Elem::Bool => ConstantScalarValue::Bool(val > 0),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicUInt => ConstantScalarValue::UInt(val),
        })
    }
    /// Create a constant scalar from a boolean.
    ///
    /// The output will have the same type as the element.
    pub fn constant_from_bool(&self, val: bool) -> Variable {
        Variable::ConstantScalar(match self {
            Elem::Float(kind) => ConstantScalarValue::Float(val as u32 as f64, *kind),
            Elem::Int(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::AtomicInt(kind) => ConstantScalarValue::Int(val as i64, *kind),
            Elem::UInt => ConstantScalarValue::UInt(val as u64),
            Elem::AtomicUInt => ConstantScalarValue::UInt(val as u64),
            Elem::Bool => ConstantScalarValue::Bool(val),
        })
    }

    /// Ensure that the variable provided, when a constant, is the same type as elem.
    pub fn from_constant(&self, constant: Variable) -> Variable {
        let value = match constant {
            Variable::ConstantScalar(value) => value,
            _ => return constant,
        };

        match value {
            ConstantScalarValue::Int(val, _) => self.constant_from_i64(val),
            ConstantScalarValue::Float(val, _) => self.constant_from_f64(val),
            ConstantScalarValue::UInt(val) => self.constant_from_u64(val),
            ConstantScalarValue::Bool(val) => self.constant_from_bool(val),
        }
    }
    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        match self {
            Elem::Float(kind) => match kind {
                FloatKind::F16 => core::mem::size_of::<half::f16>(),
                FloatKind::BF16 => core::mem::size_of::<half::bf16>(),
                FloatKind::F32 => core::mem::size_of::<f32>(),
                FloatKind::F64 => core::mem::size_of::<f64>(),
            },
            Elem::Int(kind) => match kind {
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            Elem::AtomicInt(kind) => match kind {
                IntKind::I32 => core::mem::size_of::<i32>(),
                IntKind::I64 => core::mem::size_of::<i64>(),
            },
            Elem::UInt => core::mem::size_of::<u32>(),
            Elem::AtomicUInt => core::mem::size_of::<u32>(),
            Elem::Bool => core::mem::size_of::<bool>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, Elem::AtomicInt(_) | Elem::AtomicUInt)
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
                FloatKind::F32 => f.write_str("f32"),
                FloatKind::F64 => f.write_str("f64"),
            },
            Self::Int(kind) => match kind {
                IntKind::I32 => f.write_str("i32"),
                IntKind::I64 => f.write_str("i64"),
            },
            Self::AtomicInt(kind) => match kind {
                IntKind::I32 => f.write_str("atomic<i32>"),
                IntKind::I64 => f.write_str("atomic<i64>"),
            },
            Self::UInt => f.write_str("uint"),
            Self::AtomicUInt => f.write_str("atomic<uint>"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize, Hash)]
pub struct Item {
    pub elem: Elem,
    pub vectorization: Vectorization,
}

impl Item {
    /// Fetch the elem of the item.
    pub fn elem(&self) -> Elem {
        self.elem
    }

    /// Create a new item without vectorization
    pub fn new(elem: Elem) -> Self {
        Self {
            elem,
            vectorization: 1,
        }
    }

    /// Create a new item with vectorization
    pub fn vectorized(elem: Elem, vectorization: Vectorization) -> Self {
        Self {
            elem,
            vectorization,
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
}

#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub struct CubeDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CubeDim {
    pub(crate) fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for CubeDim {
    fn default() -> Self {
        Self {
            x: SUBCUBE_DIM_APPROX as u32,
            y: SUBCUBE_DIM_APPROX as u32,
            z: 1,
        }
    }
}
