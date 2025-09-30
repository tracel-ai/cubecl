use core::fmt::Display;

use crate::{TypeHash, UnaryOperator};

use crate::{BinaryOperator, OperationReflect};

/// Comparison operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = ComparisonOpCode, pure)]
pub enum Comparison {
    Lower(BinaryOperator),
    LowerEqual(BinaryOperator),
    #[operation(commutative)]
    Equal(BinaryOperator),
    #[operation(commutative)]
    NotEqual(BinaryOperator),
    GreaterEqual(BinaryOperator),
    Greater(BinaryOperator),
    IsNan(UnaryOperator),
    IsInf(UnaryOperator),
}

impl Display for Comparison {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Comparison::Equal(op) => write!(f, "{} == {}", op.lhs, op.rhs),
            Comparison::NotEqual(op) => write!(f, "{} != {}", op.lhs, op.rhs),
            Comparison::Lower(op) => write!(f, "{} < {}", op.lhs, op.rhs),
            Comparison::Greater(op) => write!(f, "{} > {}", op.lhs, op.rhs),
            Comparison::LowerEqual(op) => write!(f, "{} <= {}", op.lhs, op.rhs),
            Comparison::GreaterEqual(op) => write!(f, "{} >= {}", op.lhs, op.rhs),
            Comparison::IsNan(op) => write!(f, "{}.isnan()", op.input),
            Comparison::IsInf(op) => write!(f, "{}.isinf()", op.input),
        }
    }
}
