use core::fmt::Display;

use crate::TypeHash;

use crate::{BinaryOperands, OperationReflect, UnaryOperands};

/// Bitwise operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = BitwiseOpCode, pure)]
pub enum Bitwise {
    #[operation(commutative)]
    BitwiseAnd(BinaryOperands),
    #[operation(commutative)]
    BitwiseOr(BinaryOperands),
    #[operation(commutative)]
    BitwiseXor(BinaryOperands),
    ShiftLeft(BinaryOperands),
    ShiftRight(BinaryOperands),
    CountOnes(UnaryOperands),
    ReverseBits(UnaryOperands),
    BitwiseNot(UnaryOperands),
    /// Count leading zeros
    LeadingZeros(UnaryOperands),
    /// Count trailing zeros
    TrailingZeros(UnaryOperands),
    /// Find least significant bit set
    FindFirstSet(UnaryOperands),
}

impl Display for Bitwise {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Bitwise::BitwiseAnd(op) => write!(f, "{} & {}", op.lhs, op.rhs),
            Bitwise::BitwiseOr(op) => write!(f, "{} | {}", op.lhs, op.rhs),
            Bitwise::BitwiseXor(op) => write!(f, "{} ^ {}", op.lhs, op.rhs),
            Bitwise::CountOnes(op) => write!(f, "{}.count_bits()", op.input),
            Bitwise::ReverseBits(op) => write!(f, "{}.reverse_bits()", op.input),
            Bitwise::ShiftLeft(op) => write!(f, "{} << {}", op.lhs, op.rhs),
            Bitwise::ShiftRight(op) => write!(f, "{} >> {}", op.lhs, op.rhs),
            Bitwise::BitwiseNot(op) => write!(f, "!{}", op.input),
            Bitwise::LeadingZeros(op) => write!(f, "{}.leading_zeros()", op.input),
            Bitwise::TrailingZeros(op) => write!(f, "{}.trailing_zeros()", op.input),
            Bitwise::FindFirstSet(op) => write!(f, "{}.find_first_set()", op.input),
        }
    }
}
