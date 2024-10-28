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
            Self::Operator(operator) => write!(f, "{operator}"),
            Self::Metadata(metadata) => write!(f, "{metadata}"),
            Self::Branch(branch) => write!(f, "{branch}"),
            Self::Synchronization(synchronization) => write!(f, "{synchronization}"),
            Self::Subcube(subcube) => write!(f, "{subcube}"),
            Self::CoopMma(coop_mma) => write!(f, "{coop_mma}"),
        }
    }
}

impl Operation {
    pub fn out(&self) -> Option<Variable> {
        match self {
            Self::Operator(operator) => operator.out(),
            Self::Metadata(metadata) => metadata.out(),
            Self::Branch(Branch::Select(op)) => Some(op.out),
            Self::Branch(_) => None,
            Self::Synchronization(_) => None,
            Self::Subcube(subcube) => subcube.out(),
            Self::CoopMma(_) => None,
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
        match self {
            Self::Add(binary_operator)
            | Self::Sub(binary_operator)
            | Self::Mul(binary_operator)
            | Self::Div(binary_operator)
            | Self::Powf(binary_operator)
            | Self::Equal(binary_operator)
            | Self::NotEqual(binary_operator)
            | Self::Lower(binary_operator)
            | Self::Greater(binary_operator)
            | Self::LowerEqual(binary_operator)
            | Self::GreaterEqual(binary_operator)
            | Self::Modulo(binary_operator)
            | Self::Index(binary_operator)
            | Self::UncheckedIndex(binary_operator)
            | Self::IndexAssign(binary_operator)
            | Self::UncheckedIndexAssign(binary_operator)
            | Self::Max(binary_operator)
            | Self::Min(binary_operator)
            | Self::BitwiseAnd(binary_operator)
            | Self::BitwiseOr(binary_operator)
            | Self::BitwiseXor(binary_operator)
            | Self::ShiftLeft(binary_operator)
            | Self::ShiftRight(binary_operator)
            | Self::Remainder(binary_operator)
            | Self::And(binary_operator)
            | Self::Or(binary_operator)
            | Self::AtomicSwap(binary_operator)
            | Self::AtomicAdd(binary_operator)
            | Self::AtomicSub(binary_operator)
            | Self::AtomicMax(binary_operator)
            | Self::AtomicMin(binary_operator)
            | Self::AtomicAnd(binary_operator)
            | Self::AtomicOr(binary_operator)
            | Self::AtomicXor(binary_operator)
            | Self::Dot(binary_operator) => binary_operator.out,

            Self::Abs(unary_operator)
            | Self::Exp(unary_operator)
            | Self::Log(unary_operator)
            | Self::Log1p(unary_operator)
            | Self::Cos(unary_operator)
            | Self::Sin(unary_operator)
            | Self::Tanh(unary_operator)
            | Self::Sqrt(unary_operator)
            | Self::Round(unary_operator)
            | Self::Floor(unary_operator)
            | Self::Ceil(unary_operator)
            | Self::Erf(unary_operator)
            | Self::Recip(unary_operator)
            | Self::Assign(unary_operator)
            | Self::Not(unary_operator)
            | Self::Neg(unary_operator)
            | Self::Bitcast(unary_operator)
            | Self::AtomicLoad(unary_operator)
            | Self::AtomicStore(unary_operator)
            | Self::Magnitude(unary_operator)
            | Self::Normalize(unary_operator) => unary_operator.out,

            Self::Clamp(clamp_operator) => clamp_operator.out,
            Self::Copy(copy_operator) => copy_operator.out,
            Self::CopyBulk(copy_bulk_operator) => copy_bulk_operator.out,
            Self::Slice(slice_operator) => slice_operator.out,
            Self::InitLine(line_init_operator) => line_init_operator.out,
            Self::AtomicCompareAndSwap(op) => op.out,
            Self::Fma(fma_operator) => fma_operator.out,
        }
        .into()
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add(op) => write!(f, "{} = {} + {}", op.out, op.lhs, op.rhs),
            Self::Fma(op) => write!(f, "{} = {} * {} + {}", op.out, op.a, op.b, op.c),
            Self::Sub(op) => write!(f, "{} = {} - {}", op.out, op.lhs, op.rhs),
            Self::Mul(op) => write!(f, "{} = {} * {}", op.out, op.lhs, op.rhs),
            Self::Div(op) => write!(f, "{} = {} / {}", op.out, op.lhs, op.rhs),
            Self::Abs(op) => write!(f, "{} = {}.abs()", op.out, op.input),
            Self::Exp(op) => write!(f, "{} = {}.exp()", op.out, op.input),
            Self::Log(op) => write!(f, "{} = {}.log()", op.out, op.input),
            Self::Log1p(op) => write!(f, "{} = {}.log_1p()", op.out, op.input),
            Self::Cos(op) => write!(f, "{} = {}.cos()", op.out, op.input),
            Self::Sin(op) => write!(f, "{} = {}.sin()", op.out, op.input),
            Self::Tanh(op) => write!(f, "{} = {}.tanh()", op.out, op.input),
            Self::Powf(op) => write!(f, "{} = {}.pow({})", op.out, op.lhs, op.rhs),
            Self::Sqrt(op) => write!(f, "{} = {}.sqrt()", op.out, op.input),
            Self::Round(op) => write!(f, "{} = {}.round()", op.out, op.input),
            Self::Floor(op) => write!(f, "{} = {}.floor()", op.out, op.input),
            Self::Ceil(op) => write!(f, "{} = {}.ceil()", op.out, op.input),
            Self::Erf(op) => write!(f, "{} = {}.erf()", op.out, op.input),
            Self::Recip(op) => write!(f, "{} = {}.recip()", op.out, op.input),
            Self::Equal(op) => write!(f, "{} = {} == {}", op.out, op.lhs, op.rhs),
            Self::NotEqual(op) => write!(f, "{} = {} != {}", op.out, op.lhs, op.rhs),
            Self::Lower(op) => write!(f, "{} = {} < {}", op.out, op.lhs, op.rhs),
            Self::Clamp(op) => write!(
                f,
                "{} = {}.clamp({}, {})",
                op.out, op.input, op.min_value, op.max_value
            ),
            Self::Greater(op) => write!(f, "{} = {} > {}", op.out, op.lhs, op.rhs),
            Self::LowerEqual(op) => write!(f, "{} = {} <= {}", op.out, op.lhs, op.rhs),
            Self::GreaterEqual(op) => write!(f, "{} = {} >= {}", op.out, op.lhs, op.rhs),
            Self::Assign(op) => write!(f, "{} = {}", op.out, op.input),
            Self::Modulo(op) => write!(f, "{} = {} % {}", op.out, op.lhs, op.rhs),
            Self::Index(op) => write!(f, "{} = {}[{}]", op.out, op.lhs, op.rhs),
            Self::Copy(op) => write!(
                f,
                "{}[{}] = {}[{}]",
                op.out, op.out_index, op.input, op.in_index
            ),
            Self::CopyBulk(op) => write!(
                f,
                "memcpy({}[{}], {}[{}], {})",
                op.out, op.input, op.in_index, op.out_index, op.len
            ),
            Self::Slice(op) => write!(f, "{} = {}[{}..{}]", op.out, op.input, op.start, op.end),
            Self::UncheckedIndex(op) => {
                write!(f, "{} = unchecked {}[{}]", op.out, op.lhs, op.rhs)
            }
            Self::IndexAssign(op) => write!(f, "{}[{}] = {}", op.out, op.lhs, op.rhs),
            Self::UncheckedIndexAssign(op) => {
                write!(f, "unchecked {}[{}] = {}", op.out, op.lhs, op.rhs)
            }
            Self::And(op) => write!(f, "{} = {} && {}", op.out, op.lhs, op.rhs),
            Self::Or(op) => write!(f, "{} = {} || {}", op.out, op.lhs, op.rhs),
            Self::Not(op) => write!(f, "{} = !{}", op.out, op.input),
            Self::Neg(op) => write!(f, "{} = -{}", op.out, op.input),
            Self::Max(op) => write!(f, "{} = {}.max({})", op.out, op.lhs, op.rhs),
            Self::Min(op) => write!(f, "{} = {}.min({})", op.out, op.lhs, op.rhs),
            Self::BitwiseAnd(op) => write!(f, "{} = {} & {}", op.out, op.lhs, op.rhs),
            Self::BitwiseOr(op) => write!(f, "{} = {} | {}", op.out, op.lhs, op.rhs),
            Self::BitwiseXor(op) => write!(f, "{} = {} ^ {}", op.out, op.lhs, op.rhs),
            Self::ShiftLeft(op) => write!(f, "{} = {} << {}", op.out, op.lhs, op.rhs),
            Self::ShiftRight(op) => write!(f, "{} = {} >> {}", op.out, op.lhs, op.rhs),
            Self::Remainder(op) => write!(f, "{} = {} rem {}", op.out, op.lhs, op.rhs),
            Self::Bitcast(op) => write!(f, "{} = bitcast({})", op.out, op.input),
            Self::AtomicLoad(op) => write!(f, "{} = atomic_load({})", op.out, op.input),
            Self::AtomicStore(op) => write!(f, "atomic_store({}, {})", op.out, op.input),
            Self::AtomicSwap(op) => {
                write!(f, "{} = atomic_swap({}, {})", op.out, op.lhs, op.rhs)
            }
            Self::AtomicAdd(op) => write!(f, "{} = atomic_add({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicSub(op) => write!(f, "{} = atomic_sub({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicMax(op) => write!(f, "{} = atomic_max({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicMin(op) => write!(f, "{} = atomic_min({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicAnd(op) => write!(f, "{} = atomic_and({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicOr(op) => write!(f, "{} = atomic_or({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicXor(op) => write!(f, "{} = atomic_xor({}, {})", op.out, op.lhs, op.rhs),
            Self::AtomicCompareAndSwap(op) => write!(
                f,
                "{} = compare_and_swap({}, {}, {})",
                op.out, op.input, op.cmp, op.val
            ),
            Self::Magnitude(op) => write!(f, "{} = {}.length()", op.out, op.input),
            Self::Normalize(op) => write!(f, "{} = {}.normalize()", op.out, op.input),
            Self::Dot(op) => write!(f, "{} = {}.dot({})", op.out, op.lhs, op.rhs),
            Self::InitLine(init) => {
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
        match self {
            Self::Stride { out, .. } | Self::Shape { out, .. } | Self::Length { out, .. } => *out,
        }
        .into()
    }
}

impl Display for Metadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stride { dim, var, out } => write!(f, "{} = {}.strides[{}]", out, var, dim),
            Self::Shape { dim, var, out } => write!(f, "{} = {}.shape[{}]", out, var, dim),
            Self::Length { var, out } => write!(f, "{} = {}.len()", out, var),
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
        Self::Operator(val)
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
        Self::Metadata(val)
    }
}
