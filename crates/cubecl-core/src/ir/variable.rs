use std::fmt::Display;
use std::num::NonZero;

use crate::prelude::CubePrimitive;

use super::{Elem, FloatKind, IntKind, Item, Matrix, UIntKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Variable {
    pub kind: VariableKind,
    pub item: Item,
}

impl Variable {
    pub fn new(kind: VariableKind, item: Item) -> Self {
        Self { kind, item }
    }

    pub fn builtin(builtin: Builtin) -> Self {
        Self::new(VariableKind::Builtin(builtin), Item::new(Elem::UInt))
    }

    pub fn constant(scalar: ConstantScalarValue) -> Self {
        let elem = match scalar {
            ConstantScalarValue::Int(_, int_kind) => Elem::Int(int_kind),
            ConstantScalarValue::Float(_, float_kind) => Elem::Float(float_kind),
            ConstantScalarValue::UInt(_) => Elem::UInt,
            ConstantScalarValue::Bool(_) => Elem::Bool,
        };
        Self::new(VariableKind::ConstantScalar(scalar), Item::new(elem))
    }
}

type Id = u16;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum VariableKind {
    GlobalInputArray(Id),
    GlobalOutputArray(Id),
    GlobalScalar(Id),
    Local { id: Id, depth: u8 },
    Versioned { id: Id, depth: u8, version: u16 },
    LocalBinding { id: Id, depth: u8 },
    ConstantScalar(ConstantScalarValue),
    ConstantArray { id: Id, length: u32 },
    SharedMemory { id: Id, length: u32 },
    LocalArray { id: Id, depth: u8, length: u32 },
    Matrix { id: Id, mat: Matrix, depth: u8 },
    Slice { id: Id, depth: u8 },
    Builtin(Builtin),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Builtin {
    Rank,
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    SubcubeDim,
    AbsolutePos,
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
}

impl Variable {
    /// Whether a variable is always immutable. Used for optimizations to determine whether it's
    /// safe to inline/merge
    pub fn is_immutable(&self) -> bool {
        match self.kind {
            VariableKind::GlobalOutputArray { .. } => false,
            VariableKind::Local { .. } => false,
            VariableKind::SharedMemory { .. } => false,
            VariableKind::Matrix { .. } => false,
            VariableKind::Slice { .. } => false,
            VariableKind::LocalArray { .. } => false,
            VariableKind::GlobalInputArray { .. } => false,
            VariableKind::GlobalScalar { .. } => true,
            VariableKind::Versioned { .. } => true,
            VariableKind::LocalBinding { .. } => true,
            VariableKind::ConstantScalar(_) => true,
            VariableKind::ConstantArray { .. } => true,
            VariableKind::Builtin(_) => true,
        }
    }

    /// Is this an array type that yields [`Item`]s when indexed, or a scalar/vector that yields
    /// [`Elem`]s when indexed?
    pub fn is_array(&self) -> bool {
        matches!(
            self.kind,
            VariableKind::GlobalInputArray { .. }
                | VariableKind::GlobalOutputArray { .. }
                | VariableKind::ConstantArray { .. }
                | VariableKind::SharedMemory { .. }
                | VariableKind::LocalArray { .. }
                | VariableKind::Matrix { .. }
                | VariableKind::Slice { .. }
        )
    }

    /// Determines if the value is a constant with the specified value (converted if necessary)
    pub fn is_constant(&self, value: i64) -> bool {
        match self.kind {
            VariableKind::ConstantScalar(ConstantScalarValue::Int(val, _)) => val == value,
            VariableKind::ConstantScalar(ConstantScalarValue::UInt(val, _)) => val as i64 == value,
            VariableKind::ConstantScalar(ConstantScalarValue::Float(val, _)) => val == value as f64,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `true` value
    pub fn is_true(&self) -> bool {
        match self.kind {
            VariableKind::ConstantScalar(ConstantScalarValue::Bool(val)) => val,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `false` value
    pub fn is_false(&self) -> bool {
        match self.kind {
            VariableKind::ConstantScalar(ConstantScalarValue::Bool(val)) => !val,
            _ => false,
        }
    }
}

/// The scalars are stored with the highest precision possible, but they might get reduced during
/// compilation.
#[derive(Debug, Clone, PartialEq, Copy, Serialize, Deserialize, PartialOrd)]
#[allow(missing_docs)]
pub enum ConstantScalarValue {
    Int(i64, IntKind),
    Float(f64, FloatKind),
    UInt(u64, UIntKind),
    Bool(bool),
}

impl ConstantScalarValue {
    /// Returns the element type of the scalar.
    pub fn elem(&self) -> Elem {
        match self {
            ConstantScalarValue::Int(_, kind) => Elem::Int(*kind),
            ConstantScalarValue::Float(_, kind) => Elem::Float(*kind),
            ConstantScalarValue::UInt(_) => Elem::UInt,
            ConstantScalarValue::Bool(_) => Elem::Bool,
        }
    }

    /// Returns the value of the scalar as a usize.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            ConstantScalarValue::UInt(val) => Some(*val as usize),
            ConstantScalarValue::Int(val, _) => Some(*val as usize),
            ConstantScalarValue::Float(_, _) => None,
            ConstantScalarValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a usize.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_usize(&self) -> usize {
        self.try_as_usize()
            .expect("Only Int and UInt kind can be made into usize.")
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u32(&self) -> Option<u32> {
        match self {
            ConstantScalarValue::UInt(val) => Some(*val as u32),
            ConstantScalarValue::Int(val, _) => Some(*val as u32),
            ConstantScalarValue::Float(_, _) => None,
            ConstantScalarValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_u32(&self) -> u32 {
        self.try_as_u32()
            .expect("Only Int and UInt kind can be made into u32.")
    }

    /// Returns the value of the scalar as a u64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u64(&self) -> Option<u64> {
        match self {
            ConstantScalarValue::UInt(val) => Some(*val),
            ConstantScalarValue::Int(val, _) => Some(*val as u64),
            ConstantScalarValue::Float(_, _) => None,
            ConstantScalarValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u64.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_u64(&self) -> u64 {
        self.try_as_u64()
            .expect("Only Int and UInt kind can be made into u64.")
    }

    /// Returns the value of the scalar as a i64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_i64(&self) -> Option<i64> {
        match self {
            ConstantScalarValue::UInt(val) => Some(*val as i64),
            ConstantScalarValue::Int(val, _) => Some(*val),
            ConstantScalarValue::Float(_, _) => None,
            ConstantScalarValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_i64(&self) -> i64 {
        self.try_as_i64()
            .expect("Only Int and UInt kind can be made into i64.")
    }

    /// Returns the value of the variable as a bool if it actually is a bool.
    pub fn try_as_bool(&self) -> Option<bool> {
        match self {
            ConstantScalarValue::Bool(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the value of the variable as a bool.
    ///
    /// It will panic if the scalar isn't a bool.
    pub fn as_bool(&self) -> bool {
        self.try_as_bool()
            .expect("Only bool can be made into a bool")
    }

    pub fn is_zero(&self) -> bool {
        match self {
            ConstantScalarValue::Int(val, _) => *val == 0,
            ConstantScalarValue::Float(val, _) => *val == 0.0,
            ConstantScalarValue::UInt(val) => *val == 0,
            ConstantScalarValue::Bool(_) => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            ConstantScalarValue::Int(val, _) => *val == 1,
            ConstantScalarValue::Float(val, _) => *val == 1.0,
            ConstantScalarValue::UInt(val) => *val == 1,
            ConstantScalarValue::Bool(_) => false,
        }
    }

    pub fn cast_to(&self, other: Elem) -> ConstantScalarValue {
        match (self, other) {
            (ConstantScalarValue::Int(val, _), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val as f64, float_kind)
            }
            (ConstantScalarValue::Int(val, _), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val, int_kind)
            }
            (ConstantScalarValue::Int(val, _), Elem::UInt) => {
                ConstantScalarValue::UInt(*val as u64)
            }
            (ConstantScalarValue::Int(val, _), Elem::Bool) => ConstantScalarValue::Bool(*val == 1),
            (ConstantScalarValue::Float(val, _), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val, float_kind)
            }
            (ConstantScalarValue::Float(val, _), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::Float(val, _), Elem::UInt) => {
                ConstantScalarValue::UInt(*val as u64)
            }
            (ConstantScalarValue::Float(val, _), Elem::Bool) => {
                ConstantScalarValue::Bool(*val == 0.0)
            }
            (ConstantScalarValue::UInt(val), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val as f64, float_kind)
            }
            (ConstantScalarValue::UInt(val), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::UInt(val), Elem::UInt) => ConstantScalarValue::UInt(*val),
            (ConstantScalarValue::UInt(val), Elem::Bool) => ConstantScalarValue::Bool(*val == 1),
            (ConstantScalarValue::Bool(val), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val as u32 as f64, float_kind)
            }
            (ConstantScalarValue::Bool(val), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::Bool(val), Elem::UInt) => ConstantScalarValue::UInt(*val as u64),
            (ConstantScalarValue::Bool(val), Elem::Bool) => ConstantScalarValue::Bool(*val),
            _ => unreachable!(),
        }
    }
}

impl Display for ConstantScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantScalarValue::Int(val, IntKind::I8) => write!(f, "{val}i8"),
            ConstantScalarValue::Int(val, IntKind::I16) => write!(f, "{val}i16"),
            ConstantScalarValue::Int(val, IntKind::I32) => write!(f, "{val}i32"),
            ConstantScalarValue::Int(val, IntKind::I64) => write!(f, "{val}i64"),
            ConstantScalarValue::Float(val, FloatKind::BF16) => write!(f, "{val}bf16"),
            ConstantScalarValue::Float(val, FloatKind::F16) => write!(f, "{val}f16"),
            ConstantScalarValue::Float(val, FloatKind::F32) => write!(f, "{val}f32"),
            ConstantScalarValue::Float(val, FloatKind::F64) => write!(f, "{val}f64"),
            ConstantScalarValue::UInt(val) => write!(f, "{val}u32"),
            ConstantScalarValue::Bool(val) => write!(f, "{val}"),
        }
    }
}

impl Variable {
    pub fn vectorization_factor(&self) -> u8 {
        self.item.vectorization.map(NonZero::get).unwrap_or(1u8)
    }
    pub fn index(&self) -> Option<u16> {
        match self.kind {
            VariableKind::GlobalInputArray(id) => Some(id),
            VariableKind::GlobalScalar(id) => Some(id),
            VariableKind::Local { id, .. } => Some(id),
            VariableKind::Versioned { id, .. } => Some(id),
            VariableKind::LocalBinding { id, .. } => Some(id),
            VariableKind::Slice { id, .. } => Some(id),
            VariableKind::GlobalOutputArray(id) => Some(id),
            VariableKind::ConstantScalar(_) => None,
            VariableKind::ConstantArray { id, .. } => Some(id),
            VariableKind::SharedMemory { id, .. } => Some(id),
            VariableKind::LocalArray { id, .. } => Some(id),
            VariableKind::Matrix { id, .. } => Some(id),
            VariableKind::Builtin(_) => None,
        }
    }

    pub fn as_const(&self) -> Option<ConstantScalarValue> {
        match self.kind {
            VariableKind::ConstantScalar(constant) => Some(constant),
            _ => None,
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            VariableKind::GlobalInputArray(id) => write!(f, "input({id})"),
            VariableKind::GlobalOutputArray(id) => write!(f, "output({id})"),
            VariableKind::GlobalScalar(id) => write!(f, "scalar({id})"),
            VariableKind::ConstantScalar(constant) => write!(f, "{constant}"),
            VariableKind::Local { id, depth } => write!(f, "local({id}, {depth})"),
            VariableKind::Versioned { id, depth, version } => {
                write!(f, "local({id}, {depth}).v{version}")
            }
            VariableKind::LocalBinding { id, depth } => write!(f, "binding({id}, {depth})"),
            VariableKind::ConstantArray { id, .. } => write!(f, "const_array({id})"),
            VariableKind::SharedMemory { id, .. } => write!(f, "shared({id})"),
            VariableKind::LocalArray { id, .. } => write!(f, "array({id})"),
            VariableKind::Matrix { id, depth, .. } => write!(f, "matrix({id}, {depth})"),
            VariableKind::Slice { id, depth } => write!(f, "slice({id}, {depth})"),
            VariableKind::Builtin(builtin) => write!(f, "{builtin:?}"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
