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
        let val = match self {
            Subcube::Elect(init_operator) => init_operator.out,
            Subcube::Broadcast(binary_operator) => binary_operator.out,
            Subcube::All(unary_operator)
            | Subcube::Any(unary_operator)
            | Subcube::Sum(unary_operator)
            | Subcube::Prod(unary_operator)
            | Subcube::Min(unary_operator)
            | Subcube::Max(unary_operator) => unary_operator.out,
        };
        Some(val)
    }
}

impl Display for Subcube {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subcube::Elect(op) => writeln!(f, "{} = subcube_elect()", op.out),
            Subcube::All(op) => writeln!(f, "{} = subcube_all({})", op.out, op.input),
            Subcube::Any(op) => writeln!(f, "{} = subcube_any({})", op.out, op.input),
            Subcube::Broadcast(op) => {
                writeln!(f, "{} = subcube_broadcast({}, {})", op.out, op.lhs, op.rhs)
            }
            Subcube::Sum(op) => writeln!(f, "{} = subcube_sum({})", op.out, op.input),
            Subcube::Prod(op) => writeln!(f, "{} = subcube_product({})", op.out, op.input),
            Subcube::Min(op) => writeln!(f, "{} = subcube_min({})", op.out, op.input),
            Subcube::Max(op) => writeln!(f, "{} = subcube_max({})", op.out, op.input),
        }
    }
}
