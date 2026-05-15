use core::fmt::Display;

use crate::TypeHash;

use crate::{BinaryOperands, OperationArgs, OperationReflect, UnaryOperands, Variable};

/// Arithmetic operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = ArithmeticOpCode, pure)]
pub enum Arithmetic {
    #[operation(commutative)]
    Add(BinaryOperands),
    #[operation(commutative)]
    SaturatingAdd(BinaryOperands),
    Fma(FmaOperands),
    Sub(BinaryOperands),
    SaturatingSub(BinaryOperands),
    #[operation(commutative)]
    Mul(BinaryOperands),
    Div(BinaryOperands),
    Abs(UnaryOperands),
    Exp(UnaryOperands),
    Log(UnaryOperands),
    Log1p(UnaryOperands),
    Cos(UnaryOperands),
    Sin(UnaryOperands),
    Tan(UnaryOperands),
    Tanh(UnaryOperands),
    Sinh(UnaryOperands),
    Cosh(UnaryOperands),
    ArcCos(UnaryOperands),
    ArcSin(UnaryOperands),
    ArcTan(UnaryOperands),
    ArcSinh(UnaryOperands),
    ArcCosh(UnaryOperands),
    ArcTanh(UnaryOperands),
    Degrees(UnaryOperands),
    Radians(UnaryOperands),
    ArcTan2(BinaryOperands),
    Powf(BinaryOperands),
    Powi(BinaryOperands),
    Hypot(BinaryOperands),
    Rhypot(BinaryOperands),
    Sqrt(UnaryOperands),
    InverseSqrt(UnaryOperands),
    Round(UnaryOperands),
    Floor(UnaryOperands),
    Ceil(UnaryOperands),
    Trunc(UnaryOperands),
    Erf(UnaryOperands),
    Recip(UnaryOperands),
    Clamp(ClampOperands),
    Neg(UnaryOperands),
    #[operation(commutative)]
    Max(BinaryOperands),
    #[operation(commutative)]
    Min(BinaryOperands),
    /// Rust `Rem::rem`
    Rem(BinaryOperands),
    /// Pytorch %, or mod in SPIR-V
    ModFloor(BinaryOperands),
    Magnitude(UnaryOperands),
    Normalize(UnaryOperands),
    #[operation(commutative)]
    Dot(BinaryOperands),
    #[operation(commutative)]
    MulHi(BinaryOperands),
    VectorSum(UnaryOperands),
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
            Arithmetic::Tan(op) => write!(f, "{}.tan()", op.input),
            Arithmetic::Tanh(op) => write!(f, "{}.tanh()", op.input),
            Arithmetic::Sinh(op) => write!(f, "{}.sinh()", op.input),
            Arithmetic::Cosh(op) => write!(f, "{}.cosh()", op.input),
            Arithmetic::ArcCos(op) => write!(f, "{}.acos()", op.input),
            Arithmetic::ArcSin(op) => write!(f, "{}.asin()", op.input),
            Arithmetic::ArcTan(op) => write!(f, "{}.atan()", op.input),
            Arithmetic::ArcSinh(op) => write!(f, "{}.asinh()", op.input),
            Arithmetic::ArcCosh(op) => write!(f, "{}.acosh()", op.input),
            Arithmetic::ArcTanh(op) => write!(f, "{}.atanh()", op.input),
            Arithmetic::Degrees(op) => write!(f, "{}.degrees()", op.input),
            Arithmetic::Radians(op) => write!(f, "{}.radians()", op.input),
            Arithmetic::ArcTan2(op) => write!(f, "{}.atan2({})", op.lhs, op.rhs),
            Arithmetic::Powf(op) => write!(f, "{}.pow({})", op.lhs, op.rhs),
            Arithmetic::Powi(op) => write!(f, "{}.powi({})", op.lhs, op.rhs),
            Arithmetic::Hypot(op) => write!(f, "{}.hypot({})", op.lhs, op.rhs),
            Arithmetic::Rhypot(op) => write!(f, "{}.rhypot({})", op.lhs, op.rhs),
            Arithmetic::Sqrt(op) => write!(f, "{}.sqrt()", op.input),
            Arithmetic::InverseSqrt(op) => write!(f, "{}.inverse_sqrt()", op.input),
            Arithmetic::Round(op) => write!(f, "{}.round()", op.input),
            Arithmetic::Floor(op) => write!(f, "{}.floor()", op.input),
            Arithmetic::Ceil(op) => write!(f, "{}.ceil()", op.input),
            Arithmetic::Trunc(op) => write!(f, "{}.trunc()", op.input),
            Arithmetic::Erf(op) => write!(f, "{}.erf()", op.input),
            Arithmetic::Recip(op) => write!(f, "{}.recip()", op.input),
            Arithmetic::Clamp(op) => {
                write!(f, "{}.clamp({}, {})", op.input, op.min_value, op.max_value)
            }
            Arithmetic::Neg(op) => write!(f, "-{}", op.input),
            Arithmetic::Max(op) => write!(f, "{}.max({})", op.lhs, op.rhs),
            Arithmetic::Min(op) => write!(f, "{}.min({})", op.lhs, op.rhs),
            Arithmetic::Rem(op) => write!(f, "{} % {}", op.lhs, op.rhs),
            Arithmetic::ModFloor(op) => write!(f, "{}.mod_floor({})", op.lhs, op.rhs),
            Arithmetic::Magnitude(op) => write!(f, "{}.length()", op.input),
            Arithmetic::Normalize(op) => write!(f, "{}.normalize()", op.input),
            Arithmetic::Dot(op) => write!(f, "{}.dot({})", op.lhs, op.rhs),
            Arithmetic::MulHi(op) => write!(f, "mul_hi({}, {})", op.lhs, op.rhs),
            Arithmetic::VectorSum(op) => write!(f, "{}.vector_sum()", op.input),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct ClampOperands {
    pub input: Variable,
    pub min_value: Variable,
    pub max_value: Variable,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationArgs)]
#[allow(missing_docs)]
pub struct FmaOperands {
    pub a: Variable,
    pub b: Variable,
    pub c: Variable,
}
