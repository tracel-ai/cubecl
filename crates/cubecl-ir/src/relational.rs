use core::fmt::Display;

use crate::TypeHash;

use crate::{OperationReflect, UnaryOperator};

/// Float relational operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = RelationalOpCode, pure)]
pub enum Relational {
    IsNan(UnaryOperator),
}

impl Display for Relational {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Relational::IsNan(op) => write!(f, "{}.isnan()", op.input),
        }
    }
}
