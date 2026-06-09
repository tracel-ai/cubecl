use core::fmt::Display;

use alloc::{format, vec::Vec};

use crate::TypeHash;

use crate::{BinaryOperands, OperationArgs, OperationReflect, UnaryOperands, Variable};

/// Operators available on the GPU
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = OperatorOpCode)]
pub enum Operator {
    #[operation(pure)]
    InitVector(InitVectorOperands),
    #[operation(pure)]
    ExtractComponent(BinaryOperands),
    #[operation(pure)]
    InsertComponent(VectorInsertOperands),
    #[operation(commutative, pure)]
    And(BinaryOperands),
    #[operation(commutative, pure)]
    Or(BinaryOperands),
    #[operation(pure)]
    Not(UnaryOperands),
    #[operation(pure)]
    Cast(UnaryOperands),
    #[operation(pure)]
    Reinterpret(UnaryOperands),
    /// A select statement/ternary
    #[operation(pure)]
    Select(SelectOperands),
}

impl Display for Operator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Operator::And(op) => write!(f, "{} && {}", op.lhs, op.rhs),
            Operator::Or(op) => write!(f, "{} || {}", op.lhs, op.rhs),
            Operator::Not(op) => write!(f, "!{}", op.input),
            Operator::InitVector(init) => {
                let inits = init
                    .inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                write!(f, "vec({})", inits.join(", "))
            }
            Operator::ExtractComponent(op) => write!(f, "{}[{}]", op.lhs, op.rhs),
            Operator::InsertComponent(op) => {
                write!(f, "{}[{}] = {}", op.vector, op.index, op.value)
            }
            Operator::Select(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.then, op.or_else)
            }
            Operator::Cast(op) => write!(f, "cast({})", op.input),
            Operator::Reinterpret(op) => write!(f, "reinterpret({})", op.input),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct SliceOperands {
    pub input: Variable,
    pub start: Variable,
    pub end: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ReinterpretSliceOperands {
    pub input: Variable,
    pub vector_size: u32,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct InitVectorOperands {
    pub inputs: Vec<Variable>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct VectorInsertOperands {
    pub vector: Variable,
    pub index: Variable,
    pub value: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryOperands {
    #[args(allow_ptr, ptr_read)]
    pub source: Variable,
    #[args(allow_ptr, ptr_write)]
    pub target: Variable,
    pub len: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct SelectOperands {
    pub cond: Variable,
    pub then: Variable,
    pub or_else: Variable,
}
