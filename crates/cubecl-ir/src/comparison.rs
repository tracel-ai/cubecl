use core::fmt::Display;

use crate::{TypeHash, UnaryOperands};

use crate::{BinaryOperands, OperationReflect};

/// Comparison operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = ComparisonOpCode, pure)]
pub enum Comparison {
    Lower(BinaryOperands),
    LowerEqual(BinaryOperands),
    #[operation(commutative)]
    Equal(BinaryOperands),
    #[operation(commutative)]
    NotEqual(BinaryOperands),
    GreaterEqual(BinaryOperands),
    Greater(BinaryOperands),
    IsNan(UnaryOperands),
    IsInf(UnaryOperands),
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
