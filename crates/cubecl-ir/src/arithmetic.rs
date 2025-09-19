use core::fmt::Display;

use crate::TypeHash;

use crate::{BinaryOperator, OperationArgs, OperationReflect, UnaryOperator, Variable};

/// Arithmetic operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = ArithmeticOpCode, pure)]
pub enum Arithmetic {
    #[operation(commutative)]
    Add(BinaryOperator),
    #[operation(commutative)]
    SaturatingAdd(BinaryOperator),
    Fma(FmaOperator),
    Sub(BinaryOperator),
    SaturatingSub(BinaryOperator),
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
    Powi(BinaryOperator),
    Sqrt(UnaryOperator),
    Round(UnaryOperator),
    Floor(UnaryOperator),
    Ceil(UnaryOperator),
    Erf(UnaryOperator),
    Recip(UnaryOperator),
    Clamp(ClampOperator),
    Modulo(BinaryOperator),
    Neg(UnaryOperator),
    #[operation(commutative)]
    Max(BinaryOperator),
    #[operation(commutative)]
    Min(BinaryOperator),
    Remainder(BinaryOperator),
    Magnitude(UnaryOperator),
    Normalize(UnaryOperator),
    #[operation(commutative)]
    Dot(BinaryOperator),
    #[operation(commutative)]
    MulHi(BinaryOperator),
}

impl Display for Arithmetic {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Arithmetic::Add(op) => write!(f, "{} + {}", op.lhs, op.rhs),
            Arithmetic::SaturatingAdd(op) => write!(f, "saturating_add({}, {})", op.lhs, op.rhs),
            Arithmetic::Fma(op) => write!(f, "{} * {} + {}", op.a, op.b, op.c),
            Arithmetic::Sub(op) => write!(f, "{} - {}", op.lhs, op.rhs),
            Arithmetic::SaturatingSub(op) => write!(f, "saturating_sub({}, {})", op.lhs, op.rhs),
            Arithmetic::Mul(op) => write!(f, "{} * {}", op.lhs, op.rhs),
            Arithmetic::Div(op) => write!(f, "{} / {}", op.lhs, op.rhs),
            Arithmetic::Abs(op) => write!(f, "{}.abs()", op.input),
            Arithmetic::Exp(op) => write!(f, "{}.exp()", op.input),
            Arithmetic::Log(op) => write!(f, "{}.log()", op.input),
            Arithmetic::Log1p(op) => write!(f, "{}.log_1p()", op.input),
            Arithmetic::Cos(op) => write!(f, "{}.cos()", op.input),
            Arithmetic::Sin(op) => write!(f, "{}.sin()", op.input),
            Arithmetic::Tanh(op) => write!(f, "{}.tanh()", op.input),
            Arithmetic::Powf(op) => write!(f, "{}.powf({})", op.lhs, op.rhs),
            Arithmetic::Powi(op) => write!(f, "{}.powi({})", op.lhs, op.rhs),
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
            Arithmetic::Neg(op) => write!(f, "-{}", op.input),
            Arithmetic::Max(op) => write!(f, "{}.max({})", op.lhs, op.rhs),
            Arithmetic::Min(op) => write!(f, "{}.min({})", op.lhs, op.rhs),
            Arithmetic::Remainder(op) => write!(f, "{} rem {}", op.lhs, op.rhs),
            Arithmetic::Magnitude(op) => write!(f, "{}.length()", op.input),
            Arithmetic::Normalize(op) => write!(f, "{}.normalize()", op.input),
            Arithmetic::Dot(op) => write!(f, "{}.dot({})", op.lhs, op.rhs),
            Arithmetic::MulHi(op) => write!(f, "mul_hi({}, {})", op.lhs, op.rhs),
        }
    }
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
