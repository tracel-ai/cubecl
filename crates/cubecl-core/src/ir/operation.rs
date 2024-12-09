use std::fmt::Display;

use super::{Branch, CoopMma, Item, Plane, Scope, Select, Synchronization, Variable};
use crate::{
    cpa,
    ir::{Elem, UIntKind},
    prelude::AtomicOp,
};
use serde::{Deserialize, Serialize};

/// All operations that can be used in a GPU compute shader.
///
/// Notes:
///
/// [Operator] can be vectorized, but other operations can't.
/// Therefore, during tracing, only operators can be registered.
///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs, clippy::large_enum_variant)] // Some variants might not be used with different flags
pub enum Operation {
    Copy(Variable),
    Operator(Operator),
    Atomic(AtomicOp),
    Metadata(Metadata),
    Branch(Branch),
    Synchronization(Synchronization),
    Plane(Plane),
    CoopMma(CoopMma),
}

/// An instruction that contains a right hand side [`Operation`] and an optional out variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
            Operation::Copy(variable) => write!(f, "{variable}"),
        }
    }
}

/// All operators that can be used in a GPU compute shader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Operator {
    Add(BinaryOperator),
    Fma(FmaOperator),
    Sub(BinaryOperator),
    Mul(BinaryOperator),
    Div(BinaryOperator),
    Abs(UnaryOperator),
    Exp(UnaryOperator),
    Log(UnaryOperator),
    Log1p(UnaryOperator),
    Cos(UnaryOperator),
    Sin(UnaryOperator),
    Tanh(UnaryOperator),
    Powf(BinaryOperator),
    Sqrt(UnaryOperator),
    Round(UnaryOperator),
    Floor(UnaryOperator),
    Ceil(UnaryOperator),
    Erf(UnaryOperator),
    Recip(UnaryOperator),
    Equal(BinaryOperator),
    NotEqual(BinaryOperator),
    Lower(BinaryOperator),
    Clamp(ClampOperator),
    Greater(BinaryOperator),
    LowerEqual(BinaryOperator),
    GreaterEqual(BinaryOperator),
    Cast(UnaryOperator),
    Modulo(BinaryOperator),
    Index(BinaryOperator),
    CopyMemory(CopyMemoryOperator),
    CopyMemoryBulk(CopyMemoryBulkOperator),
    Slice(SliceOperator),
    Ptr(UnaryOperator),
    UncheckedIndex(BinaryOperator),
    IndexAssign(BinaryOperator),
    InitLine(LineInitOperator),
    UncheckedIndexAssign(BinaryOperator),
    And(BinaryOperator),
    Or(BinaryOperator),
    Not(UnaryOperator),
    Neg(UnaryOperator),
    Max(BinaryOperator),
    Min(BinaryOperator),
    BitwiseAnd(BinaryOperator),
    BitwiseOr(BinaryOperator),
    BitwiseXor(BinaryOperator),
    ShiftLeft(BinaryOperator),
    ShiftRight(BinaryOperator),
    Remainder(BinaryOperator),
    Bitcast(UnaryOperator),
    Magnitude(UnaryOperator),
    Normalize(UnaryOperator),
    Dot(BinaryOperator),
    // A select statement/ternary
    Select(Select),
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add(op) => write!(f, "{} + {}", op.lhs, op.rhs),
            Operator::Fma(op) => write!(f, "{} * {} + {}", op.a, op.b, op.c),
            Operator::Sub(op) => write!(f, "{} - {}", op.lhs, op.rhs),
            Operator::Mul(op) => write!(f, "{} * {}", op.lhs, op.rhs),
            Operator::Div(op) => write!(f, "{} / {}", op.lhs, op.rhs),
            Operator::Abs(op) => write!(f, "{}.abs()", op.input),
            Operator::Exp(op) => write!(f, "{}.exp()", op.input),
            Operator::Log(op) => write!(f, "{}.log()", op.input),
            Operator::Log1p(op) => write!(f, "{}.log_1p()", op.input),
            Operator::Cos(op) => write!(f, "{}.cos()", op.input),
            Operator::Sin(op) => write!(f, "{}.sin()", op.input),
            Operator::Tanh(op) => write!(f, "{}.tanh()", op.input),
            Operator::Powf(op) => write!(f, "{}.pow({})", op.lhs, op.rhs),
            Operator::Sqrt(op) => write!(f, "{}.sqrt()", op.input),
            Operator::Round(op) => write!(f, "{}.round()", op.input),
            Operator::Floor(op) => write!(f, "{}.floor()", op.input),
            Operator::Ceil(op) => write!(f, "{}.ceil()", op.input),
            Operator::Erf(op) => write!(f, "{}.erf()", op.input),
            Operator::Recip(op) => write!(f, "{}.recip()", op.input),
            Operator::Equal(op) => write!(f, "{} == {}", op.lhs, op.rhs),
            Operator::NotEqual(op) => write!(f, "{} != {}", op.lhs, op.rhs),
            Operator::Lower(op) => write!(f, "{} < {}", op.lhs, op.rhs),
            Operator::Clamp(op) => {
                write!(f, "{}.clamp({}, {})", op.input, op.min_value, op.max_value)
            }
            Operator::Greater(op) => write!(f, "{} > {}", op.lhs, op.rhs),
            Operator::LowerEqual(op) => write!(f, "{} <= {}", op.lhs, op.rhs),
            Operator::GreaterEqual(op) => write!(f, "{} >= {}", op.lhs, op.rhs),
            Operator::Modulo(op) => write!(f, "{} % {}", op.lhs, op.rhs),
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
            Operator::Neg(op) => write!(f, "-{}", op.input),
            Operator::Max(op) => write!(f, "{}.max({})", op.lhs, op.rhs),
            Operator::Min(op) => write!(f, "{}.min({})", op.lhs, op.rhs),
            Operator::BitwiseAnd(op) => write!(f, "{} & {}", op.lhs, op.rhs),
            Operator::BitwiseOr(op) => write!(f, "{} | {}", op.lhs, op.rhs),
            Operator::BitwiseXor(op) => write!(f, "{} ^ {}", op.lhs, op.rhs),
            Operator::ShiftLeft(op) => write!(f, "{} << {}", op.lhs, op.rhs),
            Operator::ShiftRight(op) => write!(f, "{} >> {}", op.lhs, op.rhs),
            Operator::Remainder(op) => write!(f, "{} rem {}", op.lhs, op.rhs),
            Operator::Magnitude(op) => write!(f, "{}.length()", op.input),
            Operator::Normalize(op) => write!(f, "{}.normalize()", op.input),
            Operator::Dot(op) => write!(f, "{}.dot({})", op.lhs, op.rhs),
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
            Operator::Ptr(op) => write!(f, "ptr({})", op.input),
        }
    }
}

/// All metadata that can be accessed in a shader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Metadata {
    /// The rank of an array.
    Rank { var: Variable },
    /// The stride of an array at the given dimension.
    Stride { dim: Variable, var: Variable },
    /// The shape of an array at the given dimension.
    Shape { dim: Variable, var: Variable },
    /// The length of an array.
    Length { var: Variable },
    /// The length of an array's underlying buffer.
    BufferLength { var: Variable },
}

impl Display for Metadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metadata::Rank { var } => write!(f, "rank({})", var),
            Metadata::Stride { dim, var } => write!(f, "{}.strides[{}]", var, dim),
            Metadata::Shape { dim, var } => write!(f, "{}.shape[{}]", var, dim),
            Metadata::Length { var } => write!(f, "{}.len()", var),
            Metadata::BufferLength { var } => write!(f, "buffer_len({})", var),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct BinaryOperator {
    pub lhs: Variable,
    pub rhs: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct UnaryOperator {
    pub input: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct LineInitOperator {
    pub inputs: Vec<Variable>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CopyMemoryOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CopyMemoryBulkOperator {
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
    pub len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct ClampOperator {
    pub input: Variable,
    pub min_value: Variable,
    pub max_value: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct SliceOperator {
    pub input: Variable,
    pub start: Variable,
    pub end: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CompareAndSwapOperator {
    pub input: Variable,
    pub cmp: Variable,
    pub val: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct ReadGlobalOperator {
    pub variable: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct ReadGlobalWithLayoutOperator {
    pub variable: Variable,
    pub tensor_read_pos: usize,
    pub tensor_layout_pos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct FmaOperator {
    pub a: Variable,
    pub b: Variable,
    pub c: Variable,
}

#[allow(missing_docs)]
pub struct CheckedIndexAssign {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

impl CheckedIndexAssign {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let lhs = self.lhs;
        let rhs = self.rhs;
        let out = self.out;
        let array_len = scope.create_local(Item::new(Elem::UInt(UIntKind::U32)));
        let inside_bound = scope.create_local(Item::new(Elem::Bool));

        if out.has_buffer_length() {
            cpa!(scope, array_len = buffer_len(out));
        } else {
            cpa!(scope, array_len = len(out));
        }

        cpa!(scope, inside_bound = lhs < array_len);

        cpa!(scope, if(inside_bound).then(|scope| {
            cpa!(scope, unchecked(out[lhs]) = rhs);
        }));
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
