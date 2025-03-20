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
    Index(BinaryOperator),
    CopyMemory(CopyMemoryOperator),
    CopyMemoryBulk(CopyMemoryBulkOperator),
    #[operation(pure)]
    Slice(SliceOperator),
    #[operation(pure)]
    UncheckedIndex(BinaryOperator),
    IndexAssign(BinaryOperator),
    #[operation(pure)]
    InitLine(LineInitOperator),
    UncheckedIndexAssign(BinaryOperator),
    #[operation(commutative, pure)]
    And(BinaryOperator),
    #[operation(commutative, pure)]
    Or(BinaryOperator),
    #[operation(pure)]
    Not(UnaryOperator),
    #[operation(pure)]
    Cast(UnaryOperator),
    #[operation(pure)]
    Bitcast(UnaryOperator),
    /// A select statement/ternary
    #[operation(pure)]
    Select(Select),
    ConditionalRead(ConditionalRead),
}

impl Display for Operator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Operator::Index(op) => write!(f, "{}[{}]", op.lhs, op.rhs),
            Operator::CopyMemory(op) => {
                write!(f, "[{}] = {}[{}]", op.out_index, op.input, op.in_index)
            }
            Operator::CopyMemoryBulk(op) => write!(
                f,
                "memcpy([{}], {}[{}], {})",
                op.input, op.in_index, op.out_index, op.len
            ),
            Operator::Slice(op) => write!(f, "{}[{}..{}]", op.input, op.start, op.end),
            Operator::UncheckedIndex(op) => {
                write!(f, "unchecked {}[{}]", op.lhs, op.rhs)
            }
            Operator::IndexAssign(op) => write!(f, "[{}] = {}", op.lhs, op.rhs),
            Operator::UncheckedIndexAssign(op) => {
                write!(f, "unchecked [{}] = {}", op.lhs, op.rhs)
            }
            Operator::And(op) => write!(f, "{} && {}", op.lhs, op.rhs),
            Operator::Or(op) => write!(f, "{} || {}", op.lhs, op.rhs),
            Operator::Not(op) => write!(f, "!{}", op.input),
            Operator::InitLine(init) => {
                let inits = init
                    .inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                write!(f, "vec({})", inits.join(", "))
            }
            Operator::Select(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.then, op.or_else)
            }
            Operator::Cast(op) => write!(f, "cast({})", op.input),
            Operator::Bitcast(op) => write!(f, "bitcast({})", op.input),
            Operator::ConditionalRead(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.slice, op.fallback)
            }
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
pub struct LineInitOperator {
    pub inputs: Vec<Variable>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryBulkOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
    pub len: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct Select {
    pub cond: Variable,
    pub then: Variable,
    pub or_else: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ConditionalRead {
    pub cond: Variable,
    pub slice: Variable,
    pub index: Variable,
    pub fallback: Variable,
}
