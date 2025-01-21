use std::fmt::Display;

use super::{Branch, CoopMma, Item, NonSemantic, PipelineOps, Plane, Synchronization, Variable};
use crate::{AtomicOp, Metadata, OperationCore, Operator};
use type_hash::TypeHash;

/// All operations that can be used in a GPU compute shader.
///
/// Notes:
///
/// [Operator] can be vectorized, but other operations can't.
/// Therefore, during tracing, only operators can be registered.
///
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCore)]
#[operation(opcode_name = OpCode)]
#[allow(dead_code, missing_docs, clippy::large_enum_variant)] // Some variants might not be used with different flags
pub enum Operation {
    Copy(Variable),
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
    Pipeline(PipelineOps),
    /// Non-semantic instructions (i.e. comments, debug info)
    #[operation(nested)]
    NonSemantic(NonSemantic),
}

/// An instruction that contains a right hand side [`Operation`] and an optional out variable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct Instruction {
    pub out: Option<Variable>,
    pub operation: Operation,
}

impl Instruction {
    pub fn new(operation: impl Into<Operation>, out: Variable) -> Self {
        Instruction {
            out: Some(out),
            operation: operation.into(),
        }
    }

    pub fn out(&self) -> Variable {
        self.out.unwrap()
    }

    pub fn item(&self) -> Item {
        self.out().item
    }
}

impl Operation {
    /// Whether this operation is pure, aka has no side effects. Pure operations can be removed
    /// if their output is not needed, impure operations must be kept since their execution can
    /// affect things down the line. e.g. atomics.
    ///
    /// Operations that operate across multiple units are always considered impure.
    pub fn is_pure(&self) -> bool {
        match self {
            Operation::Copy(_) => true,
            Operation::Operator(_) => true,
            Operation::Atomic(_) => false,
            Operation::Metadata(_) => true,
            Operation::Branch(_) => false,
            Operation::Synchronization(_) => false,
            Operation::Plane(_) => false,
            Operation::CoopMma(_) => false,
            Operation::NonSemantic(_) => false,
            Operation::Pipeline(_) => false,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
                write!(f, "{}[{}] = {}", self.out(), op.lhs, op.rhs)
            }
            Operation::Operator(Operator::UncheckedIndexAssign(op)) => {
                write!(f, "unchecked {}[{}] = {}", self.out(), op.lhs, op.rhs)
            }
            Operation::Operator(Operator::Cast(op)) => {
                write!(f, "{} = cast<{}>({})", self.out(), self.item(), op.input)
            }
            Operation::Operator(Operator::Bitcast(op)) => {
                write!(f, "{} = bitcast<{}>({})", self.out(), self.item(), op.input)
            }
            _ => {
                if let Some(out) = self.out {
                    write!(f, "{out} = {}", self.operation)
                } else {
                    write!(f, "{}", self.operation)
                }
            }
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Operator(operator) => write!(f, "{operator}"),
            Operation::Atomic(atomic) => write!(f, "{atomic}"),
            Operation::Metadata(metadata) => write!(f, "{metadata}"),
            Operation::Branch(branch) => write!(f, "{branch}"),
            Operation::Synchronization(synchronization) => write!(f, "{synchronization}"),
            Operation::Plane(plane) => write!(f, "{plane}"),
            Operation::CoopMma(coop_mma) => write!(f, "{coop_mma}"),
            Operation::Copy(variable) => write!(f, "{}", variable),
            Operation::NonSemantic(non_semantic) => write!(f, "{non_semantic}"),
            Operation::Pipeline(pipeline_ops) => write!(f, "{pipeline_ops}"),
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

impl From<Operator> for Operation {
    fn from(val: Operator) -> Self {
        Operation::Operator(val)
    }
}

impl From<Branch> for Operation {
    fn from(value: Branch) -> Self {
        Self::Branch(value)
    }
}

impl From<Branch> for Instruction {
    fn from(value: Branch) -> Self {
        Instruction {
            out: None,
            operation: value.into(),
        }
    }
}

impl From<Synchronization> for Operation {
    fn from(value: Synchronization) -> Self {
        Self::Synchronization(value)
    }
}

impl From<Synchronization> for Instruction {
    fn from(value: Synchronization) -> Self {
        Instruction {
            out: None,
            operation: value.into(),
        }
    }
}

impl From<Metadata> for Operation {
    fn from(val: Metadata) -> Self {
        Operation::Metadata(val)
    }
}

impl From<NonSemantic> for Operation {
    fn from(val: NonSemantic) -> Self {
        Operation::NonSemantic(val)
    }
}

impl From<NonSemantic> for Instruction {
    fn from(value: NonSemantic) -> Self {
        Instruction {
            out: None,
            operation: value.into(),
        }
    }
}
