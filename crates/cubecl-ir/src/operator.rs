use core::fmt::Display;

use alloc::{format, vec::Vec};

use crate::{Builtin, Id, TypeHash};

use crate::{BinaryOperands, OperationArgs, OperationReflect, UnaryOperands, Value};

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
    #[operation(pure)]
    ReadBuiltin(Builtin),
    #[operation(pure)]
    ReadScalar(Id),
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
            Operator::ExtractComponent(op) => write!(f, "extract({}, {})", op.lhs, op.rhs),
            Operator::InsertComponent(op) => {
                write!(f, "insert({}, {}, {})", op.vector, op.index, op.value)
            }
            Operator::Select(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.then, op.or_else)
            }
            Operator::Cast(op) => write!(f, "cast({})", op.input),
            Operator::Reinterpret(op) => write!(f, "reinterpret({})", op.input),
            Operator::ReadBuiltin(builtin) => write!(f, "read_builtin({builtin:?})"),
            Operator::ReadScalar(id) => write!(f, "read_scalar({id})"),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct SliceOperands {
    pub input: Value,
    pub start: Value,
    pub end: Value,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ReinterpretSliceOperands {
    pub input: Value,
    pub vector_size: u32,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct InitVectorOperands {
    pub inputs: Vec<Value>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct VectorInsertOperands {
    pub vector: Value,
    pub index: Value,
    pub value: Value,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct CopyMemoryOperands {
    #[args(allow_ptr, ptr_read)]
    pub source: Value,
    #[args(allow_ptr, ptr_write)]
    pub target: Value,
    pub len: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct SelectOperands {
    pub cond: Value,
    pub then: Value,
    pub or_else: Value,
}
