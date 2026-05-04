use core::fmt::Display;

use alloc::{format, vec::Vec};

use crate::TypeHash;

use crate::{BinaryOperator, OperationArgs, OperationReflect, UnaryOperator, Variable};

/// Operators available on the GPU
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = OperatorOpCode)]
pub enum Operator {
    #[operation(pure)]
    InitVector(VectorInitOperator),
    #[operation(pure)]
    ExtractComponent(BinaryOperator),
    #[operation(pure)]
    InsertComponent(VectorInsertOperator),
    #[operation(commutative, pure)]
    And(BinaryOperator),
    #[operation(commutative, pure)]
    Or(BinaryOperator),
    #[operation(pure)]
    Not(UnaryOperator),
    #[operation(pure)]
    Cast(UnaryOperator),
    #[operation(pure)]
    Reinterpret(UnaryOperator),
    /// A select statement/ternary
    #[operation(pure)]
    Select(Select),
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
pub struct SliceOperator {
    pub input: Variable,
    pub start: Variable,
    pub end: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ReinterpretSliceOperator {
    pub input: Variable,
    pub vector_size: u32,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct VectorInitOperator {
    pub inputs: Vec<Variable>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct VectorInsertOperator {
    pub vector: Variable,
    pub index: Variable,
    pub value: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryOperator {
    #[args(allow_ptr)]
    pub source: Variable,
    #[args(allow_ptr)]
    pub target: Variable,
    pub len: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct Select {
    pub cond: Variable,
    pub then: Variable,
    pub or_else: Variable,
}
