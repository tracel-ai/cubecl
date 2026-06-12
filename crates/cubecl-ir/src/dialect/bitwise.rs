use crate::pliron::prelude::*;
use crate::{
    dialect::{pure_binop, pure_unop},
    interfaces::Pure,
};

pure_binop!("bitwise.and", BitwiseAndOp);
pure_binop!("bitwise.or", BitwiseOrOp);
pure_binop!("bitwise.xor", BitwiseXorOp);
pure_binop!("bitwise.shl", ShiftLeftOp);
pure_binop!("bitwise.shr", ShiftRightOp);
pure_unop!("bitwise.not", BitwiseNotOp);
pure_unop!("bitwise.count_ones", CountOnesOp);
pure_unop!("bitwise.reverse_bits", ReverseBitsOp);
pure_unop!("bitwise.leading_zeros", LeadingZerosBitsOp);
pure_unop!("bitwise.trailing_zeros", TrailingZerosBitsOp);
pure_unop!("bitwise.find_first_set", FindFirstSetOp);
