use core::num::NonZero;
use core::{fmt::Display, hash::Hash};

use crate::{BarrierLevel, TypeHash};

use super::{Elem, FloatKind, IntKind, Item, Matrix, UIntKind};
use float_ord::FloatOrd;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
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
        Self::new(
            VariableKind::Builtin(builtin),
            Item::new(Elem::UInt(UIntKind::U32)),
        )
    }

    pub fn constant(scalar: ConstantScalarValue) -> Self {
        let elem = match scalar {
            ConstantScalarValue::Int(_, int_kind) => Elem::Int(int_kind),
            ConstantScalarValue::Float(_, float_kind) => Elem::Float(float_kind),
            ConstantScalarValue::UInt(_, kind) => Elem::UInt(kind),
            ConstantScalarValue::Bool(_) => Elem::Bool,
        };
        Self::new(VariableKind::ConstantScalar(scalar), Item::new(elem))
    }

    pub fn elem(&self) -> Elem {
        self.item.elem
    }
}

pub type Id = u32;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
pub enum VariableKind {
    GlobalInputArray(Id),
    GlobalOutputArray(Id),
    GlobalScalar(Id),
    TensorMap(Id),
    LocalArray {
        id: Id,
        length: u32,
    },
    LocalMut {
        id: Id,
    },
    LocalConst {
        id: Id,
    },
    Versioned {
        id: Id,
        version: u16,
    },
    ConstantScalar(ConstantScalarValue),
    ConstantArray {
        id: Id,
        length: u32,
    },
    SharedMemory {
        id: Id,
        length: u32,
        alignment: Option<u32>,
    },
    Matrix {
        id: Id,
        mat: Matrix,
    },
    Builtin(Builtin),
    Pipeline {
        id: Id,
        item: Item,
        num_stages: u8,
    },
    Barrier {
        id: Id,
        item: Item,
        level: BarrierLevel,
    },
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Builtin {
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePosCluster,
    CubePosClusterX,
    CubePosClusterY,
    CubePosClusterZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeClusterDim,
    CubeClusterDimX,
    CubeClusterDimY,
    CubeClusterDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    PlaneDim,
    UnitPosPlane,
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
            VariableKind::TensorMap(_) => false,
            VariableKind::LocalMut { .. } => false,
            VariableKind::SharedMemory { .. } => false,
            VariableKind::Matrix { .. } => false,
            VariableKind::LocalArray { .. } => false,
            VariableKind::GlobalInputArray { .. } => false,
            VariableKind::GlobalScalar { .. } => true,
            VariableKind::Versioned { .. } => true,
            VariableKind::LocalConst { .. } => true,
            VariableKind::ConstantScalar(_) => true,
            VariableKind::ConstantArray { .. } => true,
            VariableKind::Builtin(_) => true,
            VariableKind::Pipeline { .. } => false,
            VariableKind::Barrier { .. } => false,
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
        )
    }

    pub fn has_length(&self) -> bool {
        matches!(
            self.kind,
            VariableKind::GlobalInputArray { .. } | VariableKind::GlobalOutputArray { .. }
        )
    }

    pub fn has_buffer_length(&self) -> bool {
        matches!(
            self.kind,
            VariableKind::GlobalInputArray { .. } | VariableKind::GlobalOutputArray { .. }
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, PartialOrd)]
#[allow(missing_docs)]
pub enum ConstantScalarValue {
    Int(i64, IntKind),
    Float(f64, FloatKind),
    UInt(u64, UIntKind),
    Bool(bool),
}

impl Eq for ConstantScalarValue {}
impl Hash for ConstantScalarValue {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        core::mem::discriminant(self).hash(ra_expand_state);
        match self {
            ConstantScalarValue::Int(f0, f1) => {
                f0.hash(ra_expand_state);
                f1.hash(ra_expand_state);
            }
            ConstantScalarValue::Float(f0, f1) => {
                FloatOrd(*f0).hash(ra_expand_state);
                f1.hash(ra_expand_state);
            }
            ConstantScalarValue::UInt(f0, f1) => {
                f0.hash(ra_expand_state);
                f1.hash(ra_expand_state);
            }
            ConstantScalarValue::Bool(f0) => {
                f0.hash(ra_expand_state);
            }
        }
    }
}

impl ConstantScalarValue {
    /// Returns the element type of the scalar.
    pub fn elem(&self) -> Elem {
        match self {
            ConstantScalarValue::Int(_, kind) => Elem::Int(*kind),
            ConstantScalarValue::Float(_, kind) => Elem::Float(*kind),
            ConstantScalarValue::UInt(_, kind) => Elem::UInt(*kind),
            ConstantScalarValue::Bool(_) => Elem::Bool,
        }
    }

    /// Returns the value of the scalar as a usize.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            ConstantScalarValue::UInt(val, _) => Some(*val as usize),
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
            ConstantScalarValue::UInt(val, _) => Some(*val as u32),
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
            ConstantScalarValue::UInt(val, _) => Some(*val),
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
            ConstantScalarValue::UInt(val, _) => Some(*val as i64),
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
            ConstantScalarValue::UInt(val, _) => *val == 0,
            ConstantScalarValue::Bool(_) => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            ConstantScalarValue::Int(val, _) => *val == 1,
            ConstantScalarValue::Float(val, _) => *val == 1.0,
            ConstantScalarValue::UInt(val, _) => *val == 1,
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
            (ConstantScalarValue::Int(val, _), Elem::UInt(kind)) => {
                ConstantScalarValue::UInt(*val as u64, kind)
            }
            (ConstantScalarValue::Int(val, _), Elem::Bool) => ConstantScalarValue::Bool(*val == 1),
            (ConstantScalarValue::Float(val, _), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val, float_kind)
            }
            (ConstantScalarValue::Float(val, _), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::Float(val, _), Elem::UInt(kind)) => {
                ConstantScalarValue::UInt(*val as u64, kind)
            }
            (ConstantScalarValue::Float(val, _), Elem::Bool) => {
                ConstantScalarValue::Bool(*val == 0.0)
            }
            (ConstantScalarValue::UInt(val, _), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val as f64, float_kind)
            }
            (ConstantScalarValue::UInt(val, _), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::UInt(val, _), Elem::UInt(kind)) => {
                ConstantScalarValue::UInt(*val, kind)
            }
            (ConstantScalarValue::UInt(val, _), Elem::Bool) => ConstantScalarValue::Bool(*val == 1),
            (ConstantScalarValue::Bool(val), Elem::Float(float_kind)) => {
                ConstantScalarValue::Float(*val as u32 as f64, float_kind)
            }
            (ConstantScalarValue::Bool(val), Elem::Int(int_kind)) => {
                ConstantScalarValue::Int(*val as i64, int_kind)
            }
            (ConstantScalarValue::Bool(val), Elem::UInt(kind)) => {
                ConstantScalarValue::UInt(*val as u64, kind)
            }
            (ConstantScalarValue::Bool(val), Elem::Bool) => ConstantScalarValue::Bool(*val),
            _ => unreachable!(),
        }
    }
}

impl Display for ConstantScalarValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConstantScalarValue::Int(val, IntKind::I8) => write!(f, "{val}i8"),
            ConstantScalarValue::Int(val, IntKind::I16) => write!(f, "{val}i16"),
            ConstantScalarValue::Int(val, IntKind::I32) => write!(f, "{val}i32"),
            ConstantScalarValue::Int(val, IntKind::I64) => write!(f, "{val}i64"),
            ConstantScalarValue::Float(val, FloatKind::E2M1) => write!(f, "{val}e2m1"),
            ConstantScalarValue::Float(val, FloatKind::E2M3) => write!(f, "{val}e2m3"),
            ConstantScalarValue::Float(val, FloatKind::E3M2) => write!(f, "{val}e3m2"),
            ConstantScalarValue::Float(val, FloatKind::E4M3) => write!(f, "{val}e4m3"),
            ConstantScalarValue::Float(val, FloatKind::E5M2) => write!(f, "{val}e5m2"),
            ConstantScalarValue::Float(val, FloatKind::UE8M0) => write!(f, "{val}ue8m0"),
            ConstantScalarValue::Float(val, FloatKind::BF16) => write!(f, "{val}bf16"),
            ConstantScalarValue::Float(val, FloatKind::F16) => write!(f, "{val}f16"),
            ConstantScalarValue::Float(val, FloatKind::TF32) => write!(f, "{val}tf32"),
            ConstantScalarValue::Float(val, FloatKind::Flex32) => write!(f, "{val}flex32"),
            ConstantScalarValue::Float(val, FloatKind::F32) => write!(f, "{val}f32"),
            ConstantScalarValue::Float(val, FloatKind::F64) => write!(f, "{val}f64"),
            ConstantScalarValue::UInt(val, UIntKind::U8) => write!(f, "{val}u8"),
            ConstantScalarValue::UInt(val, UIntKind::U16) => write!(f, "{val}u16"),
            ConstantScalarValue::UInt(val, UIntKind::U32) => write!(f, "{val}u32"),
            ConstantScalarValue::UInt(val, UIntKind::U64) => write!(f, "{val}u64"),
            ConstantScalarValue::Bool(val) => write!(f, "{val}"),
        }
    }
}

impl Variable {
    pub fn vectorization_factor(&self) -> u8 {
        self.item.vectorization.map(NonZero::get).unwrap_or(1u8)
    }

    pub fn index(&self) -> Option<Id> {
        match self.kind {
            VariableKind::GlobalInputArray(id)
            | VariableKind::GlobalOutputArray(id)
            | VariableKind::TensorMap(id)
            | VariableKind::GlobalScalar(id)
            | VariableKind::LocalMut { id, .. }
            | VariableKind::Versioned { id, .. }
            | VariableKind::LocalConst { id, .. }
            | VariableKind::ConstantArray { id, .. }
            | VariableKind::SharedMemory { id, .. }
            | VariableKind::LocalArray { id, .. }
            | VariableKind::Matrix { id, .. } => Some(id),
            _ => None,
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            VariableKind::GlobalInputArray(id) => write!(f, "input({id})"),
            VariableKind::GlobalOutputArray(id) => write!(f, "output({id})"),
            VariableKind::GlobalScalar(id) => write!(f, "scalar({id})"),
            VariableKind::TensorMap(id) => write!(f, "tensor_map({id})"),
            VariableKind::ConstantScalar(constant) => write!(f, "{constant}"),
            VariableKind::LocalMut { id } => write!(f, "local({id})"),
            VariableKind::Versioned { id, version } => {
                write!(f, "local({id}).v{version}")
            }
            VariableKind::LocalConst { id } => write!(f, "binding({id})"),
            VariableKind::ConstantArray { id, .. } => write!(f, "const_array({id})"),
            VariableKind::SharedMemory { id, .. } => write!(f, "shared({id})"),
            VariableKind::LocalArray { id, .. } => write!(f, "array({id})"),
            VariableKind::Matrix { id, .. } => write!(f, "matrix({id})"),
            VariableKind::Builtin(builtin) => write!(f, "{builtin:?}"),
            VariableKind::Pipeline { id, .. } => write!(f, "pipeline({id})"),
            VariableKind::Barrier { id, .. } => write!(f, "barrier({id})"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}
