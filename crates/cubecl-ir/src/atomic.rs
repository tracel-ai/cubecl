use core::fmt::Display;

use crate::{AtomicBinaryOperator, TypeHash};

use crate::{OperationArgs, OperationReflect, UnaryOperator, Variable};

/// Operations that operate on atomics
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = AtomicOpCode)]
pub enum AtomicOp {
    Load(#[args(allow_ptr)] UnaryOperator),
    Store(UnaryOperator),
    Swap(AtomicBinaryOperator),
    Add(AtomicBinaryOperator),
    Sub(AtomicBinaryOperator),
    Max(AtomicBinaryOperator),
    Min(AtomicBinaryOperator),
    And(AtomicBinaryOperator),
    Or(AtomicBinaryOperator),
    Xor(AtomicBinaryOperator),
    CompareAndSwap(CompareAndSwapOperator),
}

impl Display for AtomicOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AtomicOp::Load(op) => write!(f, "atomic_load({})", op.input),
            AtomicOp::Store(op) => write!(f, "atomic_store({})", op.input),
            AtomicOp::Swap(op) => {
                write!(f, "atomic_swap({}, {})", op.ptr, op.value)
            }
            AtomicOp::Add(op) => write!(f, "atomic_add({}, {})", op.ptr, op.value),
            AtomicOp::Sub(op) => write!(f, "atomic_sub({}, {})", op.ptr, op.value),
            AtomicOp::Max(op) => write!(f, "atomic_max({}, {})", op.ptr, op.value),
            AtomicOp::Min(op) => write!(f, "atomic_min({}, {})", op.ptr, op.value),
            AtomicOp::And(op) => write!(f, "atomic_and({}, {})", op.ptr, op.value),
            AtomicOp::Or(op) => write!(f, "atomic_or({}, {})", op.ptr, op.value),
            AtomicOp::Xor(op) => write!(f, "atomic_xor({}, {})", op.ptr, op.value),
            AtomicOp::CompareAndSwap(op) => {
                write!(f, "compare_and_swap({}, {}, {})", op.ptr, op.cmp, op.val)
            }
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CompareAndSwapOperator {
    #[args(allow_ptr)]
    pub ptr: Variable,
    pub cmp: Variable,
    pub val: Variable,
}
