use cubecl_macros_internal::const_eval;
use pliron::r#type::TypePtr;

use crate::{
    attributes::{IndexAttr, IntAttr, UIntAttr},
    pliron::prelude::*,
    types::scalar::UIntType,
};
use crate::{
    dialect::{pure_binop, pure_unop},
    interfaces::Pure,
};

pure_binop!("bitwise.and", BitwiseAndOp);
const_eval!(BitwiseAndOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs & rhs
});

pure_binop!("bitwise.or", BitwiseOrOp);
const_eval!(BitwiseOrOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs | rhs
});

pure_binop!("bitwise.xor", BitwiseXorOp);
const_eval!(BitwiseXorOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs ^ rhs
});

pure_binop!("bitwise.shl", ShiftLeftOp);
const_eval!(ShiftLeftOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs << rhs
});

pure_binop!("bitwise.shr", ShiftRightOp);
const_eval!(ShiftRightOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs >> rhs
});

pure_unop!("bitwise.not", BitwiseNotOp);
const_eval!(BitwiseNotOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| !inp
});

fn u32_ty(ctx: &Context) -> TypePtr<UIntType> {
    UIntType::get_instance(UIntType { width: 32 }, ctx).expect("Should be present")
}

pure_unop!("bitwise.count_ones", CountOnesOp);
const_eval!(CountOnesOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| -> UIntAttr {
        UIntAttr::new(u32_ty(ctx), inp.count_ones() as u64)
    },
});

pure_unop!("bitwise.reverse_bits", ReverseBitsOp);
const_eval!(ReverseBitsOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| inp.reverse_bits()
});

pure_unop!("bitwise.leading_zeros", LeadingZerosBitsOp);
const_eval!(LeadingZerosBitsOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| -> UIntAttr {
        UIntAttr::new(u32_ty(ctx), inp.leading_zeros() as u64)
    },
});

pure_unop!("bitwise.trailing_zeros", TrailingZerosBitsOp);
const_eval!(TrailingZerosBitsOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| -> UIntAttr {
        UIntAttr::new(u32_ty(ctx), inp.trailing_zeros() as u64)
    },
});

pure_unop!("bitwise.find_first_set", FindFirstSetOp);
const_eval!(FindFirstSetOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |inp| -> UIntAttr {
        let out = if inp == 0 { 0 } else { inp.trailing_zeros() + 1 };
        UIntAttr::new(u32_ty(ctx), out as u64)
    },
});
