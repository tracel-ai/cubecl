use std::fmt::Display;

use type_hash::TypeHash;

use crate::{BinaryOperator, OperationReflect};

/// Comparison operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = ComparisonOpCode)]
pub enum Comparison {
    Lower(BinaryOperator),
    LowerEqual(BinaryOperator),
    #[operation(commutative)]
    Equal(BinaryOperator),
    #[operation(commutative)]
    NotEqual(BinaryOperator),
    GreaterEqual(BinaryOperator),
    Greater(BinaryOperator),
}

impl Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Comparison::Equal(op) => write!(f, "{} == {}", op.lhs, op.rhs),
            Comparison::NotEqual(op) => write!(f, "{} != {}", op.lhs, op.rhs),
            Comparison::Lower(op) => write!(f, "{} < {}", op.lhs, op.rhs),
            Comparison::Greater(op) => write!(f, "{} > {}", op.lhs, op.rhs),
            Comparison::LowerEqual(op) => write!(f, "{} <= {}", op.lhs, op.rhs),
            Comparison::GreaterEqual(op) => write!(f, "{} >= {}", op.lhs, op.rhs),
        }
    }
}
