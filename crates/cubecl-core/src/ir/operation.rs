use std::fmt::Display;

use super::{Branch, CoopMma, Subcube, Synchronization, Variable};
use serde::{Deserialize, Serialize};

/// All operations that can be used in a GPU compute shader.
///
/// Notes:
///
/// [Operator] and [Procedure] can be vectorized, but [Metadata] and [Branch] can't.
/// Therefore, during tracing, only operators and procedures can be registered.
///
/// [Procedure] expansions can safely use all operation variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code, missing_docs, clippy::large_enum_variant)] // Some variants might not be used with different flags
pub enum Operation {
    Operator(Operator),
    Metadata(Metadata),
    Branch(Branch),
    Synchronization(Synchronization),
    Subcube(Subcube),
    CoopMma(CoopMma),
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Operator(operator) => write!(f, "{operator}"),
            Operation::Metadata(metadata) => write!(f, "{metadata}"),
            Operation::Branch(branch) => write!(f, "{branch}"),
            Operation::Synchronization(synchronization) => write!(f, "{synchronization}"),
            Operation::Subcube(subcube) => write!(f, "{subcube}"),
            Operation::CoopMma(coop_mma) => write!(f, "{coop_mma}"),
        }
    }
}

impl Operation {
    pub fn out(&self) -> Option<Variable> {
        match self {
            Operation::Operator(operator) => operator.out(),
            Operation::Metadata(metadata) => metadata.out(),
            Operation::Branch(Branch::Select(op)) => Some(op.out),
            Operation::Branch(_) => None,
            Operation::Synchronization(_) => None,
            Operation::Subcube(subcube) => subcube.out(),
            Operation::CoopMma(_) => None,
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
    Assign(UnaryOperator),
    Modulo(BinaryOperator),
    Index(BinaryOperator),
    Copy(CopyOperator),
    CopyBulk(CopyBulkOperator),
    Slice(SliceOperator),
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
    AtomicLoad(UnaryOperator),
    AtomicStore(UnaryOperator),
    AtomicSwap(BinaryOperator),
    AtomicAdd(BinaryOperator),
    AtomicSub(BinaryOperator),
    AtomicMax(BinaryOperator),
    AtomicMin(BinaryOperator),
    AtomicAnd(BinaryOperator),
    AtomicOr(BinaryOperator),
    AtomicXor(BinaryOperator),
    AtomicCompareAndSwap(CompareAndSwapOperator),
    Magnitude(UnaryOperator),
    Normalize(UnaryOperator),
    Dot(BinaryOperator),
}

impl Operator {
    pub fn out(&self) -> Option<Variable> {
        let val = match self {
            Operator::Add(binary_operator)
            | Operator::Sub(binary_operator)
            | Operator::Mul(binary_operator)
            | Operator::Div(binary_operator)
            | Operator::Powf(binary_operator)
            | Operator::Equal(binary_operator)
            | Operator::NotEqual(binary_operator)
            | Operator::Lower(binary_operator)
            | Operator::Greater(binary_operator)
            | Operator::LowerEqual(binary_operator)
            | Operator::GreaterEqual(binary_operator)
            | Operator::Modulo(binary_operator)
            | Operator::Index(binary_operator)
            | Operator::UncheckedIndex(binary_operator)
            | Operator::IndexAssign(binary_operator)
            | Operator::UncheckedIndexAssign(binary_operator)
            | Operator::Max(binary_operator)
            | Operator::Min(binary_operator)
            | Operator::BitwiseAnd(binary_operator)
            | Operator::BitwiseOr(binary_operator)
            | Operator::BitwiseXor(binary_operator)
            | Operator::ShiftLeft(binary_operator)
            | Operator::ShiftRight(binary_operator)
            | Operator::Remainder(binary_operator)
            | Operator::And(binary_operator)
            | Operator::Or(binary_operator)
            | Operator::AtomicSwap(binary_operator)
            | Operator::AtomicAdd(binary_operator)
            | Operator::AtomicSub(binary_operator)
            | Operator::AtomicMax(binary_operator)
            | Operator::AtomicMin(binary_operator)
            | Operator::AtomicAnd(binary_operator)
            | Operator::AtomicOr(binary_operator)
            | Operator::AtomicXor(binary_operator)
            | Operator::Dot(binary_operator) => binary_operator.out,

            Operator::Abs(unary_operator)
            | Operator::Exp(unary_operator)
            | Operator::Log(unary_operator)
            | Operator::Log1p(unary_operator)
            | Operator::Cos(unary_operator)
            | Operator::Sin(unary_operator)
            | Operator::Tanh(unary_operator)
            | Operator::Sqrt(unary_operator)
            | Operator::Round(unary_operator)
            | Operator::Floor(unary_operator)
            | Operator::Ceil(unary_operator)
            | Operator::Erf(unary_operator)
            | Operator::Recip(unary_operator)
            | Operator::Assign(unary_operator)
            | Operator::Not(unary_operator)
            | Operator::Neg(unary_operator)
            | Operator::Bitcast(unary_operator)
            | Operator::AtomicLoad(unary_operator)
            | Operator::AtomicStore(unary_operator)
            | Operator::Magnitude(unary_operator)
            | Operator::Normalize(unary_operator) => unary_operator.out,

            Operator::Clamp(clamp_operator) => clamp_operator.out,
            Operator::Copy(copy_operator) => copy_operator.out,
            Operator::CopyBulk(copy_bulk_operator) => copy_bulk_operator.out,
            Operator::Slice(slice_operator) => slice_operator.out,
            Operator::InitLine(line_init_operator) => line_init_operator.out,
            Operator::AtomicCompareAndSwap(op) => op.out,
            Operator::Fma(fma_operator) => fma_operator.out,
        };
        Some(val)
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add(op) => write!(f, "{} = {} + {}", op.out, op.lhs, op.rhs),
            Operator::Fma(op) => write!(f, "{} = {} * {} + {}", op.out, op.a, op.b, op.c),
            Operator::Sub(op) => write!(f, "{} = {} - {}", op.out, op.lhs, op.rhs),
            Operator::Mul(op) => write!(f, "{} = {} * {}", op.out, op.lhs, op.rhs),
            Operator::Div(op) => write!(f, "{} = {} / {}", op.out, op.lhs, op.rhs),
            Operator::Abs(op) => write!(f, "{} = {}.abs()", op.out, op.input),
            Operator::Exp(op) => write!(f, "{} = {}.exp()", op.out, op.input),
            Operator::Log(op) => write!(f, "{} = {}.log()", op.out, op.input),
            Operator::Log1p(op) => write!(f, "{} = {}.log_1p()", op.out, op.input),
            Operator::Cos(op) => write!(f, "{} = {}.cos()", op.out, op.input),
            Operator::Sin(op) => write!(f, "{} = {}.sin()", op.out, op.input),
            Operator::Tanh(op) => write!(f, "{} = {}.tanh()", op.out, op.input),
            Operator::Powf(op) => write!(f, "{} = {}.pow({})", op.out, op.lhs, op.rhs),
            Operator::Sqrt(op) => write!(f, "{} = {}.sqrt()", op.out, op.input),
            Operator::Round(op) => write!(f, "{} = {}.round()", op.out, op.input),
            Operator::Floor(op) => write!(f, "{} = {}.floor()", op.out, op.input),
            Operator::Ceil(op) => write!(f, "{} = {}.ceil()", op.out, op.input),
            Operator::Erf(op) => write!(f, "{} = {}.erf()", op.out, op.input),
            Operator::Recip(op) => write!(f, "{} = {}.recip()", op.out, op.input),
            Operator::Equal(op) => write!(f, "{} = {} == {}", op.out, op.lhs, op.rhs),
            Operator::NotEqual(op) => write!(f, "{} = {} != {}", op.out, op.lhs, op.rhs),
            Operator::Lower(op) => write!(f, "{} = {} < {}", op.out, op.lhs, op.rhs),
            Operator::Clamp(op) => write!(
                f,
                "{} = {}.clamp({}, {})",
                op.out, op.input, op.min_value, op.max_value
            ),
            Operator::Greater(op) => write!(f, "{} = {} > {}", op.out, op.lhs, op.rhs),
            Operator::LowerEqual(op) => write!(f, "{} = {} <= {}", op.out, op.lhs, op.rhs),
            Operator::GreaterEqual(op) => write!(f, "{} = {} >= {}", op.out, op.lhs, op.rhs),
            Operator::Assign(op) => write!(f, "{} = {}", op.out, op.input),
            Operator::Modulo(op) => write!(f, "{} = {} % {}", op.out, op.lhs, op.rhs),
            Operator::Index(op) => write!(f, "{} = {}[{}]", op.out, op.lhs, op.rhs),
            Operator::Copy(op) => write!(
                f,
                "{}[{}] = {}[{}]",
                op.out, op.out_index, op.input, op.in_index
            ),
            Operator::CopyBulk(op) => write!(
                f,
                "memcpy({}[{}], {}[{}], {})",
                op.out, op.input, op.in_index, op.out_index, op.len
            ),
            Operator::Slice(op) => write!(f, "{} = {}[{}..{}]", op.out, op.input, op.start, op.end),
            Operator::UncheckedIndex(op) => {
                write!(f, "{} = unchecked {}[{}]", op.out, op.lhs, op.rhs)
            }
            Operator::IndexAssign(op) => write!(f, "{}[{}] = {}", op.out, op.lhs, op.rhs),
            Operator::UncheckedIndexAssign(op) => {
                write!(f, "unchecked {}[{}] = {}", op.out, op.lhs, op.rhs)
            }
            Operator::And(op) => write!(f, "{} = {} && {}", op.out, op.lhs, op.rhs),
            Operator::Or(op) => write!(f, "{} = {} || {}", op.out, op.lhs, op.rhs),
            Operator::Not(op) => write!(f, "{} = !{}", op.out, op.input),
            Operator::Neg(op) => write!(f, "{} = -{}", op.out, op.input),
            Operator::Max(op) => write!(f, "{} = {}.max({})", op.out, op.lhs, op.rhs),
            Operator::Min(op) => write!(f, "{} = {}.min({})", op.out, op.lhs, op.rhs),
            Operator::BitwiseAnd(op) => write!(f, "{} = {} & {}", op.out, op.lhs, op.rhs),
            Operator::BitwiseOr(op) => write!(f, "{} = {} | {}", op.out, op.lhs, op.rhs),
            Operator::BitwiseXor(op) => write!(f, "{} = {} ^ {}", op.out, op.lhs, op.rhs),
            Operator::ShiftLeft(op) => write!(f, "{} = {} << {}", op.out, op.lhs, op.rhs),
            Operator::ShiftRight(op) => write!(f, "{} = {} >> {}", op.out, op.lhs, op.rhs),
            Operator::Remainder(op) => write!(f, "{} = {} rem {}", op.out, op.lhs, op.rhs),
            Operator::Bitcast(op) => write!(f, "{} = bitcast({})", op.out, op.input),
            Operator::AtomicLoad(op) => write!(f, "{} = atomic_load({})", op.out, op.input),
            Operator::AtomicStore(op) => write!(f, "atomic_store({}, {})", op.out, op.input),
            Operator::AtomicSwap(op) => {
                write!(f, "{} = atomic_swap({}, {})", op.out, op.lhs, op.rhs)
            }
            Operator::AtomicAdd(op) => write!(f, "{} = atomic_add({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicSub(op) => write!(f, "{} = atomic_sub({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicMax(op) => write!(f, "{} = atomic_max({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicMin(op) => write!(f, "{} = atomic_min({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicAnd(op) => write!(f, "{} = atomic_and({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicOr(op) => write!(f, "{} = atomic_or({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicXor(op) => write!(f, "{} = atomic_xor({}, {})", op.out, op.lhs, op.rhs),
            Operator::AtomicCompareAndSwap(op) => write!(
                f,
                "{} = compare_and_swap({}, {}, {})",
                op.out, op.input, op.cmp, op.val
            ),
            Operator::Magnitude(op) => write!(f, "{} = {}.length()", op.out, op.input),
            Operator::Normalize(op) => write!(f, "{} = {}.normalize()", op.out, op.input),
            Operator::Dot(op) => write!(f, "{} = {}.dot({})", op.out, op.lhs, op.rhs),
            Operator::InitLine(init) => {
                let inits = init
                    .inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                write!(f, "{} = vec({})", init.out, inits.join(", "))
            }
        }
    }
}

/// All metadata that can be access in a shader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum Metadata {
    /// The stride of an array at the given dimension.
    Stride {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
    /// The shape of an array at the given dimension.
    Shape {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
    Length {
        var: Variable,
        out: Variable,
    },
}

impl Metadata {
    pub fn out(&self) -> Option<Variable> {
        let val = match self {
            Metadata::Stride { out, .. } => *out,
            Metadata::Shape { out, .. } => *out,
            Metadata::Length { out, .. } => *out,
        };
        Some(val)
    }
}

impl Display for Metadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metadata::Stride { dim, var, out } => write!(f, "{} = {}.strides[{}]", out, var, dim),
            Metadata::Shape { dim, var, out } => write!(f, "{} = {}.shape[{}]", out, var, dim),
            Metadata::Length { var, out } => write!(f, "{} = {}.len()", out, var),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct BinaryOperator {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct UnaryOperator {
    pub input: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct InitOperator {
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct LineInitOperator {
    pub out: Variable,
    pub inputs: Vec<Variable>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CopyOperator {
    pub out: Variable,
    pub out_index: Variable,
    pub input: Variable,
    pub in_index: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CopyBulkOperator {
    pub out: Variable,
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
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct SliceOperator {
    pub input: Variable,
    pub start: Variable,
    pub end: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct CompareAndSwapOperator {
    pub input: Variable,
    pub cmp: Variable,
    pub val: Variable,
    pub out: Variable,
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
    pub out: Variable,
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

impl From<Synchronization> for Operation {
    fn from(value: Synchronization) -> Self {
        Self::Synchronization(value)
    }
}

impl From<Metadata> for Operation {
    fn from(val: Metadata) -> Self {
        Operation::Metadata(val)
    }
}
