use std::fmt::Display;

use type_hash::TypeHash;

use crate::{cpa, Elem, Item, OperationArgs, OperationReflect, Scope, Select, UIntKind, Variable};

/// All operators that can be used in a GPU compute shader.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = OperatorOpCode)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Arithmetic {
    #[operation(commutative)]
    Add(BinaryOperator),
    Fma(FmaOperator),
    Sub(BinaryOperator),
    #[operation(commutative)]
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
    Clamp(ClampOperator),
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
    #[operation(commutative)]
    And(BinaryOperator),
    #[operation(commutative)]
    Or(BinaryOperator),
    Not(UnaryOperator),
    Neg(UnaryOperator),
    #[operation(commutative)]
    Max(BinaryOperator),
    #[operation(commutative)]
    Min(BinaryOperator),
    Remainder(BinaryOperator),
    Bitcast(UnaryOperator),
    Magnitude(UnaryOperator),
    Normalize(UnaryOperator),
    #[operation(commutative)]
    Dot(BinaryOperator),
    // A select statement/ternary
    Select(Select),
}

impl Display for Arithmetic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arithmetic::Add(op) => write!(f, "{} + {}", op.lhs, op.rhs),
            Arithmetic::Fma(op) => write!(f, "{} * {} + {}", op.a, op.b, op.c),
            Arithmetic::Sub(op) => write!(f, "{} - {}", op.lhs, op.rhs),
            Arithmetic::Mul(op) => write!(f, "{} * {}", op.lhs, op.rhs),
            Arithmetic::Div(op) => write!(f, "{} / {}", op.lhs, op.rhs),
            Arithmetic::Abs(op) => write!(f, "{}.abs()", op.input),
            Arithmetic::Exp(op) => write!(f, "{}.exp()", op.input),
            Arithmetic::Log(op) => write!(f, "{}.log()", op.input),
            Arithmetic::Log1p(op) => write!(f, "{}.log_1p()", op.input),
            Arithmetic::Cos(op) => write!(f, "{}.cos()", op.input),
            Arithmetic::Sin(op) => write!(f, "{}.sin()", op.input),
            Arithmetic::Tanh(op) => write!(f, "{}.tanh()", op.input),
            Arithmetic::Powf(op) => write!(f, "{}.pow({})", op.lhs, op.rhs),
            Arithmetic::Sqrt(op) => write!(f, "{}.sqrt()", op.input),
            Arithmetic::Round(op) => write!(f, "{}.round()", op.input),
            Arithmetic::Floor(op) => write!(f, "{}.floor()", op.input),
            Arithmetic::Ceil(op) => write!(f, "{}.ceil()", op.input),
            Arithmetic::Erf(op) => write!(f, "{}.erf()", op.input),
            Arithmetic::Recip(op) => write!(f, "{}.recip()", op.input),

            Arithmetic::Clamp(op) => {
                write!(f, "{}.clamp({}, {})", op.input, op.min_value, op.max_value)
            }

            Arithmetic::Modulo(op) => write!(f, "{} % {}", op.lhs, op.rhs),
            Arithmetic::Index(op) => write!(f, "{}[{}]", op.lhs, op.rhs),
            Arithmetic::CopyMemory(op) => {
                write!(f, "[{}] = {}[{}]", op.out_index, op.input, op.in_index)
            }
            Arithmetic::CopyMemoryBulk(op) => write!(
                f,
                "memcpy([{}], {}[{}], {})",
                op.input, op.in_index, op.out_index, op.len
            ),
            Arithmetic::Slice(op) => write!(f, "{}[{}..{}]", op.input, op.start, op.end),
            Arithmetic::UncheckedIndex(op) => {
                write!(f, "unchecked {}[{}]", op.lhs, op.rhs)
            }
            Arithmetic::IndexAssign(op) => write!(f, "[{}] = {}", op.lhs, op.rhs),
            Arithmetic::UncheckedIndexAssign(op) => {
                write!(f, "unchecked [{}] = {}", op.lhs, op.rhs)
            }
            Arithmetic::And(op) => write!(f, "{} && {}", op.lhs, op.rhs),
            Arithmetic::Or(op) => write!(f, "{} || {}", op.lhs, op.rhs),
            Arithmetic::Not(op) => write!(f, "!{}", op.input),
            Arithmetic::Neg(op) => write!(f, "-{}", op.input),
            Arithmetic::Max(op) => write!(f, "{}.max({})", op.lhs, op.rhs),
            Arithmetic::Min(op) => write!(f, "{}.min({})", op.lhs, op.rhs),
            Arithmetic::Remainder(op) => write!(f, "{} rem {}", op.lhs, op.rhs),
            Arithmetic::Magnitude(op) => write!(f, "{}.length()", op.input),
            Arithmetic::Normalize(op) => write!(f, "{}.normalize()", op.input),
            Arithmetic::Dot(op) => write!(f, "{}.dot({})", op.lhs, op.rhs),
            Arithmetic::InitLine(init) => {
                let inits = init
                    .inputs
                    .iter()
                    .map(|input| format!("{input}"))
                    .collect::<Vec<_>>();
                write!(f, "vec({})", inits.join(", "))
            }
            Arithmetic::Select(op) => {
                write!(f, "{} ? {} : {}", op.cond, op.then, op.or_else)
            }
            Arithmetic::Cast(op) => write!(f, "cast({})", op.input),
            Arithmetic::Bitcast(op) => write!(f, "bitcast({})", op.input),
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
