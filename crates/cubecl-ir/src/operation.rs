use core::fmt::Display;

use super::{Branch, CoopMma, NonSemantic, Plane, Synchronization, Type, Variable};
use crate::{
    Arithmetic, AtomicOp, Bitwise, Metadata, OperationArgs, OperationReflect, Operator, TmaOps,
    comparison::Comparison,
};
use crate::{BarrierOps, SourceLoc, TypeHash};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use derive_more::derive::From;

/// All operations that can be used in a GPU compute shader.
///
/// Notes:
///
/// [Operator] can be vectorized, but other operations can't.
/// Therefore, during tracing, only operators can be registered.
///
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, From, OperationReflect)]
#[operation(opcode_name = OpCode)]
#[allow(dead_code, missing_docs, clippy::large_enum_variant)] // Some variants might not be used with different flags
pub enum Operation {
    #[operation(pure)]
    #[from(ignore)]
    Copy(Variable),
    #[operation(nested)]
    Arithmetic(Arithmetic),
    #[operation(nested)]
    Comparison(Comparison),
    #[operation(nested)]
    Bitwise(Bitwise),
    #[operation(nested)]
    Operator(Operator),
    #[operation(nested)]
    Atomic(AtomicOp),
    #[operation(nested)]
    Metadata(Metadata),
    #[operation(nested)]
    Branch(Branch),
    #[operation(nested)]
    Synchronization(Synchronization),
    #[operation(nested)]
    Plane(Plane),
    #[operation(nested)]
    CoopMma(CoopMma),
    #[operation(nested)]
    Barrier(BarrierOps),
    #[operation(nested)]
    Tma(TmaOps),
    /// Non-semantic instructions (i.e. comments, debug info)
    #[operation(nested)]
    NonSemantic(NonSemantic),
    /// Frees a shared memory, allowing reuse in later blocks. Only used as a marker for the shared
    /// memory analysis, should be ignored by compilers.
    Free(Variable),
}

/// An instruction that contains a right hand side [`Operation`] and an optional out variable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct Instruction {
    pub out: Option<Variable>,
    pub source_loc: Option<SourceLoc>,
    pub operation: Operation,
}

impl Instruction {
    pub fn new(operation: impl Into<Operation>, out: Variable) -> Self {
        Instruction {
            out: Some(out),
            operation: operation.into(),
            source_loc: None,
        }
    }

    pub fn no_out(operation: impl Into<Operation>) -> Self {
        Instruction {
            out: None,
            operation: operation.into(),
            source_loc: None,
        }
    }

    pub fn out(&self) -> Variable {
        self.out.unwrap()
    }

    pub fn ty(&self) -> Type {
        self.out().ty
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.operation {
            Operation::Operator(Operator::CopyMemory(op)) => write!(
                f,
                "copy_mem({}[{}], {}[{}])",
                self.out(),
                op.out_index,
                op.input,
                op.in_index
            ),
            Operation::Operator(Operator::CopyMemoryBulk(op)) => write!(
                f,
                "copy_mem_bulk({}[{}], {}[{}], {})",
                self.out(),
                op.out_index,
                op.input,
                op.in_index,
                op.len
            ),
            Operation::Operator(Operator::IndexAssign(op)) => {
                write!(
                    f,
                    "{}[{}] = {}  : ({}, {}) -> ({})",
                    self.out(),
                    op.index,
                    op.value,
                    op.index.ty,
                    op.value.ty,
                    self.out().ty,
                )
            }
            Operation::Operator(Operator::UncheckedIndexAssign(op)) => {
                write!(
                    f,
                    "unchecked {}[{}] = {} : ({}, {}) -> ({})",
                    self.out(),
                    op.index,
                    op.value,
                    op.index.ty,
                    op.value.ty,
                    self.out().ty,
                )
            }
            Operation::Operator(Operator::Cast(op)) => {
                write!(
                    f,
                    "{} = cast<{}>({}) : ({}) -> ({})",
                    self.out(),
                    self.ty(),
                    op.input,
                    op.input.ty,
                    self.out().ty,
                )
            }
            Operation::Operator(Operator::Reinterpret(op)) => {
                write!(f, "{} = bitcast<{}>({})", self.out(), self.ty(), op.input)
            }
            _ => {
                if let Some(out) = self.out {
                    let mut vars_str = String::new();
                    for (i, var) in self.operation.args().unwrap_or_default().iter().enumerate() {
                        if i != 0 {
                            vars_str.push_str(", ");
                        }
                        vars_str.push_str(&var.ty.to_string());
                    }
                    write!(
                        f,
                        "{out} = {} : ({}) -> ({})",
                        self.operation, vars_str, out.ty
                    )
                } else {
                    write!(f, "{}", self.operation)
                }
            }
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Operation::Arithmetic(arithmetic) => write!(f, "{arithmetic}"),
            Operation::Comparison(comparison) => write!(f, "{comparison}"),
            Operation::Bitwise(bitwise) => write!(f, "{bitwise}"),
            Operation::Operator(operator) => write!(f, "{operator}"),
            Operation::Atomic(atomic) => write!(f, "{atomic}"),
            Operation::Metadata(metadata) => write!(f, "{metadata}"),
            Operation::Branch(branch) => write!(f, "{branch}"),
            Operation::Synchronization(synchronization) => write!(f, "{synchronization}"),
            Operation::Plane(plane) => write!(f, "{plane}"),
            Operation::CoopMma(coop_mma) => write!(f, "{coop_mma}"),
            Operation::Copy(variable) => write!(f, "{variable}"),
            Operation::NonSemantic(non_semantic) => write!(f, "{non_semantic}"),
            Operation::Barrier(barrier_ops) => write!(f, "{barrier_ops}"),
            Operation::Tma(tma_ops) => write!(f, "{tma_ops}"),
            Operation::Free(var) => write!(f, "free({var})"),
        }
    }
}

pub fn fmt_vararg(args: &[impl Display]) -> String {
    if args.is_empty() {
        "".to_string()
    } else {
        let str = args
            .iter()
            .map(|it| it.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(", {str}")
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct IndexOperator {
    pub list: Variable,
    pub index: Variable,
    pub line_size: u32,     // 0 == same as list.
    pub unroll_factor: u32, // Adjustment factor for bounds check
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct IndexAssignOperator {
    // list is out.
    pub index: Variable,
    pub value: Variable,
    pub line_size: u32,     // 0 == same as list.
    pub unroll_factor: u32, // Adjustment factor for bounds check
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct BinaryOperator {
    pub lhs: Variable,
    pub rhs: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct UnaryOperator {
    pub input: Variable,
}

impl From<Branch> for Instruction {
    fn from(value: Branch) -> Self {
        Instruction::no_out(value)
    }
}

impl From<Synchronization> for Instruction {
    fn from(value: Synchronization) -> Self {
        Instruction::no_out(value)
    }
}

impl From<NonSemantic> for Instruction {
    fn from(value: NonSemantic) -> Self {
        Instruction::no_out(value)
    }
}
