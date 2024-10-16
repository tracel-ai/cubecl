use std::fmt::Display;
use std::num::NonZero;

use super::{Elem, FloatKind, IntKind, Item, Matrix};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Variable {
    Rank,
    GlobalInputArray {
        id: u16,
        item: Item,
    },
    GlobalScalar {
        id: u16,
        elem: Elem,
    },
    GlobalOutputArray {
        id: u16,
        item: Item,
    },
    Local {
        id: u16,
        item: Item,
        depth: u8,
    },
    Versioned {
        id: u16,
        item: Item,
        depth: u8,
        version: u16,
    },
    LocalBinding {
        id: u16,
        item: Item,
        depth: u8,
    },
    ConstantScalar(ConstantScalarValue),
    ConstantArray {
        id: u16,
        item: Item,
        length: u32,
    },
    SharedMemory {
        id: u16,
        item: Item,
        length: u32,
    },
    LocalArray {
        id: u16,
        item: Item,
        depth: u8,
        length: u32,
    },
    Matrix {
        id: u16,
        mat: Matrix,
        depth: u8,
    },
    Slice {
        id: u16,
        item: Item,
        depth: u8,
    },
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
        match self {
            Variable::GlobalOutputArray { .. } => false,
            Variable::Local { .. } => false,
            Variable::SharedMemory { .. } => false,
            Variable::Matrix { .. } => false,
            Variable::Slice { .. } => false,
            Variable::LocalArray { .. } => false,
            Variable::GlobalInputArray { .. } => false,
            Variable::GlobalScalar { .. } => true,
            Variable::Versioned { .. } => true,
            Variable::LocalBinding { .. } => true,
            Variable::ConstantScalar(_) => true,
            Variable::ConstantArray { .. } => true,
            Variable::Rank => true,
            Variable::UnitPos => true,
            Variable::UnitPosX => true,
            Variable::UnitPosY => true,
            Variable::UnitPosZ => true,
            Variable::CubePos => true,
            Variable::CubePosX => true,
            Variable::CubePosY => true,
            Variable::CubePosZ => true,
            Variable::CubeDim => true,
            Variable::CubeDimX => true,
            Variable::CubeDimY => true,
            Variable::CubeDimZ => true,
            Variable::CubeCount => true,
            Variable::CubeCountX => true,
            Variable::CubeCountY => true,
            Variable::CubeCountZ => true,
            Variable::SubcubeDim => true,
            Variable::AbsolutePos => true,
            Variable::AbsolutePosX => true,
            Variable::AbsolutePosY => true,
            Variable::AbsolutePosZ => true,
        }
    }

    /// Is this an array type that yields [`Item`]s when indexed, or a scalar/vector that yields
    /// [`Elem`]s when indexed?
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            Variable::GlobalInputArray { .. }
                | Variable::GlobalOutputArray { .. }
                | Variable::ConstantArray { .. }
                | Variable::SharedMemory { .. }
                | Variable::LocalArray { .. }
                | Variable::Matrix { .. }
                | Variable::Slice { .. }
        )
    }

    /// Determines if the value is a constant with the specified value (converted if necessary)
    pub fn is_constant(&self, value: i64) -> bool {
        match self {
            Variable::ConstantScalar(ConstantScalarValue::Int(val, _)) => *val == value,
            Variable::ConstantScalar(ConstantScalarValue::UInt(val)) => *val as i64 == value,
            Variable::ConstantScalar(ConstantScalarValue::Float(val, _)) => *val == value as f64,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `true` value
    pub fn is_true(&self) -> bool {
        match self {
            Variable::ConstantScalar(ConstantScalarValue::Bool(val)) => *val,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `false` value
    pub fn is_false(&self) -> bool {
        match self {
            Variable::ConstantScalar(ConstantScalarValue::Bool(val)) => !(*val),
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
    UInt(u64),
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
        self.item().vectorization.map(NonZero::get).unwrap_or(1u8)
    }
    pub fn index(&self) -> Option<u16> {
        match self {
            Variable::GlobalInputArray { id, .. } => Some(*id),
            Variable::GlobalScalar { id, .. } => Some(*id),
            Variable::Local { id, .. } => Some(*id),
            Variable::Versioned { id, .. } => Some(*id),
            Variable::LocalBinding { id, .. } => Some(*id),
            Variable::Slice { id, .. } => Some(*id),
            Variable::GlobalOutputArray { id, .. } => Some(*id),
            Variable::ConstantScalar { .. } => None,
            Variable::ConstantArray { id, .. } => Some(*id),
            Variable::SharedMemory { id, .. } => Some(*id),
            Variable::LocalArray { id, .. } => Some(*id),
            Variable::Matrix { id, .. } => Some(*id),
            Variable::AbsolutePos => None,
            Variable::UnitPos => None,
            Variable::UnitPosX => None,
            Variable::UnitPosY => None,
            Variable::UnitPosZ => None,
            Variable::Rank => None,
            Variable::CubePosX => None,
            Variable::CubePosY => None,
            Variable::CubePosZ => None,
            Variable::AbsolutePosX => None,
            Variable::AbsolutePosY => None,
            Variable::AbsolutePosZ => None,
            Variable::CubeDimX => None,
            Variable::CubeDimY => None,
            Variable::CubeDimZ => None,
            Variable::CubeCountX => None,
            Variable::CubeCountY => None,
            Variable::CubeCountZ => None,
            Variable::CubePos => None,
            Variable::CubeCount => None,
            Variable::CubeDim => None,
            Variable::SubcubeDim => None,
        }
    }

    /// Fetch the item of the variable.
    pub fn item(&self) -> Item {
        match self {
            Variable::GlobalInputArray { item, .. } => *item,
            Variable::GlobalOutputArray { item, .. } => *item,
            Variable::GlobalScalar { elem, .. } => Item::new(*elem),
            Variable::Local { item, .. } => *item,
            Variable::Versioned { item, .. } => *item,
            Variable::LocalBinding { item, .. } => *item,
            Variable::ConstantScalar(value) => Item::new(value.elem()),
            Variable::ConstantArray { item, .. } => *item,
            Variable::SharedMemory { item, .. } => *item,
            Variable::LocalArray { item, .. } => *item,
            Variable::Slice { item, .. } => *item,
            Variable::Matrix { mat, .. } => Item::new(mat.elem),
            Variable::AbsolutePos => Item::new(Elem::UInt),
            Variable::Rank => Item::new(Elem::UInt),
            Variable::UnitPos => Item::new(Elem::UInt),
            Variable::UnitPosX => Item::new(Elem::UInt),
            Variable::UnitPosY => Item::new(Elem::UInt),
            Variable::UnitPosZ => Item::new(Elem::UInt),
            Variable::CubePosX => Item::new(Elem::UInt),
            Variable::CubePosY => Item::new(Elem::UInt),
            Variable::CubePosZ => Item::new(Elem::UInt),
            Variable::AbsolutePosX => Item::new(Elem::UInt),
            Variable::AbsolutePosY => Item::new(Elem::UInt),
            Variable::AbsolutePosZ => Item::new(Elem::UInt),
            Variable::CubeDimX => Item::new(Elem::UInt),
            Variable::CubeDimY => Item::new(Elem::UInt),
            Variable::CubeDimZ => Item::new(Elem::UInt),
            Variable::CubeCountX => Item::new(Elem::UInt),
            Variable::CubeCountY => Item::new(Elem::UInt),
            Variable::CubeCountZ => Item::new(Elem::UInt),
            Variable::CubePos => Item::new(Elem::UInt),
            Variable::CubeCount => Item::new(Elem::UInt),
            Variable::CubeDim => Item::new(Elem::UInt),
            Variable::SubcubeDim => Item::new(Elem::UInt),
        }
    }

    pub fn as_const(&self) -> Option<ConstantScalarValue> {
        match self {
            Variable::ConstantScalar(constant) => Some(*constant),
            _ => None,
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray { id, .. } => write!(f, "input({id})"),
            Variable::GlobalScalar { id, .. } => write!(f, "scalar({id})"),
            Variable::GlobalOutputArray { id, .. } => write!(f, "output({id})"),
            Variable::ConstantScalar(constant) => write!(f, "{constant}"),
            Variable::Local { id, depth, .. } => write!(f, "local({id}, {depth})"),
            Variable::Versioned {
                id, depth, version, ..
            } => write!(f, "local({id}, {depth}).v{version}"),
            Variable::LocalBinding { id, depth, .. } => write!(f, "binding({id}, {depth})"),
            Variable::ConstantArray { id, .. } => write!(f, "const_array({id})"),
            Variable::SharedMemory { id, .. } => write!(f, "shared({id})"),
            Variable::LocalArray { id, .. } => write!(f, "array({id})"),
            Variable::Matrix { id, depth, .. } => write!(f, "matrix({id}, {depth})"),
            Variable::Slice { id, depth, .. } => write!(f, "slice({id}, {depth})"),
            builtin => write!(f, "{builtin:?}"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
