use std::fmt::Display;

use type_hash::TypeHash;

use crate::{cpa, Elem, Item, OperationArgs, OperationCore, Scope, Select, UIntKind, Variable};

/// All operators that can be used in a GPU compute shader.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCore)]
#[operation(opcode_name = OperatorOpCode)]
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
    CountOnes(UnaryOperator),
    ReverseBits(UnaryOperator),
    BitwiseNot(UnaryOperator),
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
            Operator::CountOnes(op) => write!(f, "{}.count_bits()", op.input),
            Operator::ReverseBits(op) => write!(f, "{}.reverse_bits()", op.input),
            Operator::ShiftLeft(op) => write!(f, "{} << {}", op.lhs, op.rhs),
            Operator::ShiftRight(op) => write!(f, "{} >> {}", op.lhs, op.rhs),
            Operator::BitwiseNot(op) => write!(f, "!{}", op.input),
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
        }
    }
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
pub struct ClampOperator {
    pub input: Variable,
    pub min_value: Variable,
    pub max_value: Variable,
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
pub struct CompareAndSwapOperator {
    pub input: Variable,
    pub cmp: Variable,
    pub val: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ReadGlobalOperator {
    pub variable: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct ReadGlobalWithLayoutOperator {
    pub variable: Variable,
    pub tensor_read_pos: usize,
    pub tensor_layout_pos: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct FmaOperator {
    pub a: Variable,
    pub b: Variable,
    pub c: Variable,
}

#[allow(missing_docs)]
pub fn expand_checked_index_assign(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
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
