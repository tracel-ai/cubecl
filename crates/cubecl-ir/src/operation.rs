use core::fmt::Display;

use super::{Branch, CoopMma, NonSemantic, Plane, Synchronization, Type, Variable};
use crate::{
    Arithmetic, AtomicOp, Bitwise, Id, InstructionModes, Memory, Metadata, OperationArgs,
    OperationReflect, Operator, Scope, TensorIndexingOps, TmaOps, VectorSize,
    comparison::Comparison, marker::Marker,
};
use crate::{BarrierOps, SourceLoc, TypeHash};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use derive_more::derive::From;
use itertools::Itertools;

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
    Copy(#[args(allow_ptr)] Variable),
    /// Construct an aggregate (i.e. fat pointer) that's later disaggregated into normal variables
    /// supported by codegen. Not allowed to exist after the disaggregation pass.
    ConstructAggregate(#[args(allow_ptr)] Vec<Variable>),
    /// Extract a specific field from an aggregate (i.e. read the length from a slice ptr).
    /// Not allowed to exist after the disaggregation pass.
    ExtractAggregateField(AggregateExtractOperands),
    #[operation(nested)]
    Memory(Memory),
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
    #[operation(nested)]
    TensorIndexing(TensorIndexingOps),
    /// Non-semantic instructions (i.e. comments, debug info)
    #[operation(nested)]
    NonSemantic(NonSemantic),
    // Markers used by compilers to update state or modes, but don't emit instructions
    #[operation(nested)]
    Marker(Marker),
}

/// An instruction that contains a right hand side [`Operation`] and an optional out variable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct Instruction {
    pub out: Option<Variable>,
    pub source_loc: Option<SourceLoc>,
    pub modes: InstructionModes,
    pub operation: Operation,
}

impl Instruction {
    pub fn new(operation: impl Into<Operation>, out: Variable) -> Self {
        Instruction {
            out: Some(out),
            operation: operation.into(),
            source_loc: None,
            modes: Default::default(),
        }
    }

    pub fn no_out(operation: impl Into<Operation>) -> Self {
        Instruction {
            out: None,
            operation: operation.into(),
            source_loc: None,
            modes: Default::default(),
        }
    }

    #[track_caller]
    pub fn out(&self) -> Variable {
        self.out.unwrap()
    }

    #[track_caller]
    pub fn ty(&self) -> Type {
        self.out().ty
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.operation {
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
            Operation::Copy(variable) => write!(f, "{variable}"),
            Operation::ConstructAggregate(variables) => {
                write!(f, "aggregate({})", variables.iter().join(", "))
            }
            Operation::ExtractAggregateField(AggregateExtractOperands { aggregate, field }) => {
                write!(f, "extract({aggregate}, {field})")
            }
            Operation::Memory(memory) => write!(f, "{memory}"),
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
            Operation::NonSemantic(non_semantic) => write!(f, "{non_semantic}"),
            Operation::Barrier(barrier_ops) => write!(f, "{barrier_ops}"),
            Operation::Tma(tma_ops) => write!(f, "{tma_ops}"),
            Operation::TensorIndexing(ops) => write!(f, "{ops}"),
            Operation::Marker(marker) => write!(f, "{marker}"),
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
pub struct IndexOperands {
    pub list: Variable,
    pub index: Variable,
    pub vector_size: VectorSize, // 0 == same as list.
    pub unroll_factor: usize,    // Adjustment factor for bounds check
    pub checked: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct StoreOperands {
    #[args(allow_ptr, ptr_write)]
    pub ptr: Variable,
    pub value: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct BinaryOperands {
    pub lhs: Variable,
    pub rhs: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct AtomicBinaryOperands {
    #[args(allow_ptr, ptr_read, ptr_write)]
    pub ptr: Variable,
    pub value: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct AggregateExtractOperands {
    #[args(allow_ptr)]
    pub aggregate: Variable,
    pub field: usize,
}

/// Closure passed to an intrinsic
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
pub struct Function {
    /// Explicit parameters passed to the function. Does not contain closure captures.
    pub explicit_params: Vec<Variable>,
    /// Scope containing closure instructions. Unknown variables that aren't explicit params are
    /// assumed to be captures.
    pub scope: Scope,
}

/// Closures are functions invoked by the runtime, not us. So we only use the Id, no params.
pub type Closure = Id;

impl Display for Function {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let params = self.explicit_params.iter().join(", ");
        let instructions = self.scope.instructions.borrow().iter().join("\n");
        write!(f, "|{params}| {{\n{instructions}\n}}")
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct UnaryOperands {
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
