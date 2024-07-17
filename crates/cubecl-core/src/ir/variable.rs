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
    LocalScalar {
        id: u16,
        elem: Elem,
        depth: u8,
    },
    ConstantScalar(ConstantScalarValue),
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
    /// It will panics if the scalar type is a float or a bool.
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
    /// It will panics if the scalar type is a float or a bool.
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
    /// It will panics if the scalar type is a float or a bool.
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
            ConstantScalarValue::Int(val, _) => Some(*val as i64),
            ConstantScalarValue::Float(_, _) => None,
            ConstantScalarValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panics if the scalar type is a float or a bool.
    pub fn as_i64(&self) -> i64 {
        self.try_as_i64()
            .expect("Only Int and UInt kind can be made into i64.")
    }
}

impl Variable {
    pub fn index(&self) -> Option<u16> {
        match self {
            Variable::GlobalInputArray { id, .. } => Some(*id),
            Variable::GlobalScalar { id, .. } => Some(*id),
            Variable::Local { id, .. } => Some(*id),
            Variable::Slice { id, .. } => Some(*id),
            Variable::LocalScalar { id, .. } => Some(*id),
            Variable::GlobalOutputArray { id, .. } => Some(*id),
            Variable::ConstantScalar { .. } => None,
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
            Variable::LocalScalar { elem, .. } => Item::new(*elem),
            Variable::ConstantScalar(value) => Item::new(value.elem()),
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
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
