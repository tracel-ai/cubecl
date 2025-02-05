use core::fmt::Display;

use crate::TypeHash;

use crate::{BinaryOperator, OperationArgs, OperationReflect, UnaryOperator, Variable};

/// Operations that operate on atomics
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = AtomicOpCode)]
pub enum AtomicOp {
    Load(UnaryOperator),
    Store(UnaryOperator),
    Swap(BinaryOperator),
    Add(BinaryOperator),
    Sub(BinaryOperator),
    Max(BinaryOperator),
    Min(BinaryOperator),
    And(BinaryOperator),
    Or(BinaryOperator),
    Xor(BinaryOperator),
    CompareAndSwap(CompareAndSwapOperator),
}

impl Display for AtomicOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AtomicOp::Load(op) => write!(f, "atomic_load({})", op.input),
            AtomicOp::Store(op) => write!(f, "atomic_store({})", op.input),
            AtomicOp::Swap(op) => {
                write!(f, "atomic_swap({}, {})", op.lhs, op.rhs)
            }
            AtomicOp::Add(op) => write!(f, "atomic_add({}, {})", op.lhs, op.rhs),
            AtomicOp::Sub(op) => write!(f, "atomic_sub({}, {})", op.lhs, op.rhs),
            AtomicOp::Max(op) => write!(f, "atomic_max({}, {})", op.lhs, op.rhs),
            AtomicOp::Min(op) => write!(f, "atomic_min({}, {})", op.lhs, op.rhs),
            AtomicOp::And(op) => write!(f, "atomic_and({}, {})", op.lhs, op.rhs),
            AtomicOp::Or(op) => write!(f, "atomic_or({}, {})", op.lhs, op.rhs),
            AtomicOp::Xor(op) => write!(f, "atomic_xor({}, {})", op.lhs, op.rhs),
            AtomicOp::CompareAndSwap(op) => {
                write!(f, "compare_and_swap({}, {}, {})", op.input, op.cmp, op.val)
            }
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CompareAndSwapOperator {
    pub input: Variable,
    pub cmp: Variable,
    pub val: Variable,
}
