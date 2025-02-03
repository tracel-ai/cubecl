use core::fmt::Display;

use crate::OperationReflect;

use super::{BinaryOperator, UnaryOperator};
use crate::TypeHash;

/// All plane operations.
///
/// Note that not all backends support plane (warp/subgroup) operations. Use the [runtime flag](crate::Feature::Plane).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = PlaneOpCode)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Plane {
    Elect,
    All(UnaryOperator),
    Any(UnaryOperator),
    Ballot(UnaryOperator),
    Broadcast(BinaryOperator),
    Sum(UnaryOperator),
    InclusiveSum(UnaryOperator),
    ExclusiveSum(UnaryOperator),
    Prod(UnaryOperator),
    InclusiveProd(UnaryOperator),
    ExclusiveProd(UnaryOperator),
    Min(UnaryOperator),
    Max(UnaryOperator),
}

impl Display for Plane {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Plane::Elect => writeln!(f, "plane_elect()"),
            Plane::All(op) => writeln!(f, "plane_all({})", op.input),
            Plane::Any(op) => writeln!(f, "plane_any({})", op.input),
            Plane::Ballot(op) => writeln!(f, "plane_ballot({})", op.input),
            Plane::Broadcast(op) => {
                writeln!(f, "plane_broadcast({}, {})", op.lhs, op.rhs)
            }
            Plane::Sum(op) => writeln!(f, "plane_sum({})", op.input),
            Plane::InclusiveSum(op) => writeln!(f, "plane_inclusive_sum({})", op.input),
            Plane::ExclusiveSum(op) => writeln!(f, "plane_exclusive_sum({})", op.input),
            Plane::Prod(op) => writeln!(f, "plane_product({})", op.input),
            Plane::InclusiveProd(op) => writeln!(f, "plane_inclusive_product({})", op.input),
            Plane::ExclusiveProd(op) => writeln!(f, "plane_exclusive_product({})", op.input),
            Plane::Min(op) => writeln!(f, "plane_min({})", op.input),
            Plane::Max(op) => writeln!(f, "plane_max({})", op.input),
        }
    }
}
