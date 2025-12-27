use core::{fmt::Display, hash::Hash};

use crate::{BarrierLevel, StorageType, TypeHash};

use super::{ElemType, Matrix, Type, UIntKind};
use float_ord::FloatOrd;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct Variable {
    pub kind: VariableKind,
    pub ty: Type,
}

impl Variable {
    pub fn new(kind: VariableKind, item: Type) -> Self {
        Self { kind, ty: item }
    }

    pub fn builtin(builtin: Builtin) -> Self {
        Self::new(
            VariableKind::Builtin(builtin),
            Type::scalar(ElemType::UInt(UIntKind::U32)),
        )
    }

    pub fn constant(value: ConstantValue, ty: impl Into<Type>) -> Self {
        Self::new(VariableKind::Constant(value), ty.into())
    }

    pub fn elem_type(&self) -> ElemType {
        self.ty.elem_type()
    }

    pub fn storage_type(&self) -> StorageType {
        self.ty.storage_type()
    }
}

pub type Id = u32;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
pub enum VariableKind {
    GlobalInputArray(Id),
    GlobalOutputArray(Id),
    GlobalScalar(Id),
    TensorMapInput(Id),
    TensorMapOutput(Id),
    LocalArray {
        id: Id,
        length: u32,
        unroll_factor: u32,
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
    Constant(ConstantValue),
    ConstantArray {
        id: Id,
        length: u32,
        unroll_factor: u32,
    },
    SharedArray {
        id: Id,
        length: u32,
        unroll_factor: u32,
        alignment: Option<u32>,
    },
    Shared {
        id: Id,
    },
    Matrix {
        id: Id,
        mat: Matrix,
    },
    Builtin(Builtin),
    Pipeline {
        id: Id,
        num_stages: u8,
    },
    BarrierToken {
        id: Id,
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
            VariableKind::TensorMapInput(_) => true,
            VariableKind::TensorMapOutput(_) => false,
            VariableKind::LocalMut { .. } => false,
            VariableKind::SharedArray { .. } => false,
            VariableKind::Shared { .. } => false,
            VariableKind::Matrix { .. } => false,
            VariableKind::LocalArray { .. } => false,
            VariableKind::GlobalInputArray { .. } => false,
            VariableKind::GlobalScalar { .. } => true,
            VariableKind::Versioned { .. } => true,
            VariableKind::LocalConst { .. } => true,
            VariableKind::Constant(_) => true,
            VariableKind::ConstantArray { .. } => true,
            VariableKind::Builtin(_) => true,
            VariableKind::Pipeline { .. } => false,
            VariableKind::BarrierToken { .. } => false,
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
                | VariableKind::SharedArray { .. }
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
            VariableKind::Constant(ConstantValue::Int(val)) => val == value,
            VariableKind::Constant(ConstantValue::UInt(val)) => val as i64 == value,
            VariableKind::Constant(ConstantValue::Float(val)) => val == value as f64,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `true` value
    pub fn is_true(&self) -> bool {
        match self.kind {
            VariableKind::Constant(ConstantValue::Bool(val)) => val,
            _ => false,
        }
    }

    /// Determines if the value is a boolean constant with the `false` value
    pub fn is_false(&self) -> bool {
        match self.kind {
            VariableKind::Constant(ConstantValue::Bool(val)) => !val,
            _ => false,
        }
    }
}

/// The scalars are stored with the highest precision possible, but they might get reduced during
/// compilation. For constant propagation, casts are always executed before converting back to the
/// larger type to ensure deterministic output.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, PartialOrd)]
#[allow(missing_docs, clippy::derive_ord_xor_partial_ord)]
pub enum ConstantValue {
    Int(i64),
    Float(f64),
    UInt(u64),
    Bool(bool),
}

impl Ord for ConstantValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Override float-float comparison with `FloatOrd` since `f64` isn't `Ord`. All other
        // comparisons are safe to unwrap since they're either `Ord` or only compare discriminants.
        match (self, other) {
            (ConstantValue::Float(this), ConstantValue::Float(other)) => {
                FloatOrd(*this).cmp(&FloatOrd(*other))
            }
            _ => self.partial_cmp(other).unwrap(),
        }
    }
}

impl Eq for ConstantValue {}
impl Hash for ConstantValue {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        core::mem::discriminant(self).hash(ra_expand_state);
        match self {
            ConstantValue::Int(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Float(f0) => {
                FloatOrd(*f0).hash(ra_expand_state);
            }
            ConstantValue::UInt(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Bool(f0) => {
                f0.hash(ra_expand_state);
            }
        }
    }
}

impl ConstantValue {
    /// Returns the value of the constant as a usize.
    ///
    /// It will return [None] if the constant type is a float or a bool.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            ConstantValue::UInt(val) => Some(*val as usize),
            ConstantValue::Int(val) => Some(*val as usize),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the constant as a usize.
    pub fn as_usize(&self) -> usize {
        match self {
            ConstantValue::UInt(val) => *val as usize,
            ConstantValue::Int(val) => *val as usize,
            ConstantValue::Float(val) => *val as usize,
            ConstantValue::Bool(val) => *val as usize,
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u32(&self) -> Option<u32> {
        self.try_as_u64().map(|it| it as u32)
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }

    /// Returns the value of the scalar as a u64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u64(&self) -> Option<u64> {
        match self {
            ConstantValue::UInt(val) => Some(*val),
            ConstantValue::Int(val) => Some(*val as u64),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a u64.
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstantValue::UInt(val) => *val,
            ConstantValue::Int(val) => *val as u64,
            ConstantValue::Float(val) => *val as u64,
            ConstantValue::Bool(val) => *val as u64,
        }
    }

    /// Returns the value of the scalar as a i64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_i64(&self) -> Option<i64> {
        match self {
            ConstantValue::UInt(val) => Some(*val as i64),
            ConstantValue::Int(val) => Some(*val),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) => None,
        }
    }

    /// Returns the value of the scalar as a i64.
    pub fn as_i64(&self) -> i64 {
        match self {
            ConstantValue::UInt(val) => *val as i64,
            ConstantValue::Int(val) => *val,
            ConstantValue::Float(val) => *val as i64,
            ConstantValue::Bool(val) => *val as i64,
        }
    }

    /// Returns the value of the scalar as a f64.
    ///
    /// It will return [None] if the scalar type is an int or a bool.
    pub fn try_as_f64(&self) -> Option<f64> {
        match self {
            ConstantValue::Float(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the value of the scalar as a f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            ConstantValue::UInt(val) => *val as f64,
            ConstantValue::Int(val) => *val as f64,
            ConstantValue::Float(val) => *val,
            ConstantValue::Bool(val) => *val as u8 as f64,
        }
    }

    /// Returns the value of the variable as a bool if it actually is a bool.
    pub fn try_as_bool(&self) -> Option<bool> {
        match self {
            ConstantValue::Bool(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the value of the variable as a bool.
    ///
    /// It will panic if the scalar isn't a bool.
    pub fn as_bool(&self) -> bool {
        match self {
            ConstantValue::UInt(val) => *val != 0,
            ConstantValue::Int(val) => *val != 0,
            ConstantValue::Float(val) => *val != 0.,
            ConstantValue::Bool(val) => *val,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 0,
            ConstantValue::Float(val) => *val == 0.0,
            ConstantValue::UInt(val) => *val == 0,
            ConstantValue::Bool(val) => !*val,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 1,
            ConstantValue::Float(val) => *val == 1.0,
            ConstantValue::UInt(val) => *val == 1,
            ConstantValue::Bool(val) => *val,
        }
    }

    pub fn cast_to(&self, other: Type) -> ConstantValue {
        other.constant(*self).as_const().unwrap()
    }
}

impl Display for ConstantValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConstantValue::Int(val) => write!(f, "{val}"),
            ConstantValue::Float(val) => write!(f, "{val}"),
            ConstantValue::UInt(val) => write!(f, "{val}"),
            ConstantValue::Bool(val) => write!(f, "{val}"),
        }
    }
}

impl Variable {
    pub fn line_size(&self) -> u32 {
        self.ty.line_size()
    }

    pub fn index(&self) -> Option<Id> {
        match self.kind {
            VariableKind::GlobalInputArray(id)
            | VariableKind::GlobalOutputArray(id)
            | VariableKind::TensorMapInput(id)
            | VariableKind::TensorMapOutput(id)
            | VariableKind::GlobalScalar(id)
            | VariableKind::LocalMut { id, .. }
            | VariableKind::Versioned { id, .. }
            | VariableKind::LocalConst { id, .. }
            | VariableKind::ConstantArray { id, .. }
            | VariableKind::SharedArray { id, .. }
            | VariableKind::Shared { id, .. }
            | VariableKind::LocalArray { id, .. }
            | VariableKind::Matrix { id, .. } => Some(id),
            _ => None,
        }
    }

    pub fn as_const(&self) -> Option<ConstantValue> {
        match self.kind {
            VariableKind::Constant(constant) => Some(constant),
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
            VariableKind::TensorMapInput(id) => write!(f, "tensor_map({id})"),
            VariableKind::TensorMapOutput(id) => write!(f, "tensor_map({id})"),
            VariableKind::Constant(constant) => write!(f, "{}({constant})", self.ty),
            VariableKind::LocalMut { id } => write!(f, "local({id})"),
            VariableKind::Versioned { id, version } => {
                write!(f, "local({id}).v{version}")
            }
            VariableKind::LocalConst { id } => write!(f, "binding({id})"),
            VariableKind::ConstantArray { id, .. } => write!(f, "const_array({id})"),
            VariableKind::SharedArray { id, .. } => write!(f, "shared_array({id})"),
            VariableKind::Shared { id } => write!(f, "shared({id})"),
            VariableKind::LocalArray { id, .. } => write!(f, "array({id})"),
            VariableKind::Matrix { id, .. } => write!(f, "matrix({id})"),
            VariableKind::Builtin(builtin) => write!(f, "{builtin:?}"),
            VariableKind::Pipeline { id, .. } => write!(f, "pipeline({id})"),
            VariableKind::BarrierToken { id, .. } => write!(f, "barrier_token({id})"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&Variable> for Variable {
    fn from(value: &Variable) -> Self {
        *value
    }
}

impl<T: Into<Variable>> From<T> for ConstantValue {
    fn from(value: T) -> Self {
        value.into().as_const().unwrap()
    }
}
