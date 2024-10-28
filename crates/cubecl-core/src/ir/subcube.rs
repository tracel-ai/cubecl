use std::fmt::Display;

use super::{BinaryOperator, InitOperator, UnaryOperator, Variable};
use serde::{Deserialize, Serialize};

/// All subcube operations.
///
/// Note that not all backends support subcube (warp/subgroup) operations. Use the [runtime flag](crate::Feature::Subcube).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subcube {
    Elect(InitOperator),
    All(UnaryOperator),
    Any(UnaryOperator),
    Broadcast(BinaryOperator),
    Sum(UnaryOperator),
    Prod(UnaryOperator),
    Min(UnaryOperator),
    Max(UnaryOperator),
}

impl Subcube {
    pub fn out(&self) -> Option<Variable> {
        match self {
            Self::Elect(init_operator) => init_operator.out,
            Self::Broadcast(binary_operator) => binary_operator.out,
            Self::All(unary_operator)
            | Self::Any(unary_operator)
            | Self::Sum(unary_operator)
            | Self::Prod(unary_operator)
            | Self::Min(unary_operator)
            | Self::Max(unary_operator) => unary_operator.out,
        }
        .into()
    }
}

impl Display for Subcube {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Elect(op) => writeln!(f, "{} = subcube_elect()", op.out),
            Self::All(op) => writeln!(f, "{} = subcube_all({})", op.out, op.input),
            Self::Any(op) => writeln!(f, "{} = subcube_any({})", op.out, op.input),
            Self::Broadcast(op) => {
                writeln!(f, "{} = subcube_broadcast({}, {})", op.out, op.lhs, op.rhs)
            }
            Self::Sum(op) => writeln!(f, "{} = subcube_sum({})", op.out, op.input),
            Self::Prod(op) => writeln!(f, "{} = subcube_product({})", op.out, op.input),
            Self::Min(op) => writeln!(f, "{} = subcube_min({})", op.out, op.input),
            Self::Max(op) => writeln!(f, "{} = subcube_max({})", op.out, op.input),
        }
    }
}
