use core::fmt::Display;

use enumset::{EnumSet, EnumSetType};

use crate::{Instruction, Operation, TypeHash};

use crate::{OperationCode, OperationReflect};

use super::Variable;

/// Operations that don't change the semantics of the kernel. In other words, operations that do not
/// perform any computation, if they run at all. i.e. `println`, comments and debug symbols.
///
/// Can be safely removed or ignored without changing the kernel result.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = MarkerOpCode)]
pub enum Marker {
    /// Frees a shared memory, allowing reuse in later blocks.
    Free(Variable),
    /// Updates the `FastMath` options
    SetFastMath(EnumSet<FastMath>),
}

impl OperationReflect for Marker {
    type OpCode = MarkerOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }
}

impl Display for Marker {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Marker::Free(var) => write!(f, "free({var})"),
            Marker::SetFastMath(mode) => write!(f, "set_fast_math({mode:?})"),
        }
    }
}

impl From<Marker> for Instruction {
    fn from(value: Marker) -> Self {
        Instruction::no_out(Operation::Marker(value))
    }
}

/// Unchecked optimizations for float operations. May cause precision differences, or undefined
/// behaviour if the relevant conditions are not followed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default, Debug, Hash, TypeHash, EnumSetType)]
pub enum FastMath {
    /// Disable unsafe optimizations
    #[default]
    None,
    /// Assume values are never `NaN`. If they are, the result is considered undefined behaviour.
    NotNaN,
    /// Assume values are never `Inf`/`-Inf`. If they are, the result is considered undefined
    /// behaviour.
    NotInf,
    /// Ignore sign on zero values.
    UnsignedZero,
    /// Allow swapping float division with a reciprocal, even if that swap would change precision.
    AllowReciprocal,
    /// Allow contracting float operations into fewer operations, even if the precision could
    /// change.
    AllowContraction,
    /// Allow reassociation for float operations, even if the precision could change.
    AllowReassociation,
    /// Allow all mathematical transformations for float operations, including contraction and
    /// reassociation, even if the precision could change.
    AllowTransform,
    /// Allow using lower precision intrinsics
    ReducedPrecision,
}

impl FastMath {
    pub fn all() -> EnumSet<FastMath> {
        EnumSet::all()
    }
}
