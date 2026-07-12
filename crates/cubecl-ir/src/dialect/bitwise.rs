#![allow(clippy::redundant_guards, reason = "-1 is bugged with the macro")]

use cubecl_macros_internal::{const_eval, simplify};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        types::{IntegerType, Signedness},
    },
    r#type::TypedHandle,
    utils::apint::{APInt, bw},
};

use crate::{
    ConstantValue,
    attributes::{IndexAttr, IntAttrExt},
    dialect::{
        cmp::{is_max_uint, width},
        math::int_attr,
    },
    interfaces::TypedExt,
    prelude::*,
};
use crate::{
    dialect::{pure_binop, pure_unop},
    types::VectorType,
};

pure_binop!("bitwise.and", BitwiseAndOp);
const_eval!(BitwiseAndOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs & rhs,
    // x & 0 -> 0; 0 & x -> 0;
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        Some(match const_val.as_const_val(ctx) {
            ConstantValue::Int(0) => int_attr(ctx, const_val.get_type(ctx), 0),
            ConstantValue::UInt(0) => int_attr(ctx, const_val.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(BitwiseAndOp, {
    // -1 & x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(val) if val == -1 => {
            Some(self.rhs(ctx))
        }
        ConstantValue::UInt(val) if is_max_uint(ctx, lhs?.get_type(ctx), val) => {
            Some(self.rhs(ctx))
        }
        _ => None?,
    },
    // x & -1 -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(val) if val == -1 => {
            Some(self.lhs(ctx))
        }
        ConstantValue::UInt(val) if is_max_uint(ctx, rhs?.get_type(ctx), val) => {
            Some(self.lhs(ctx))
        }
        _ => None?,
    },
    // x & x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("bitwise.or", BitwiseOrOp);
const_eval!(BitwiseOrOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs | rhs
});
simplify!(BitwiseOrOp, {
    // 0 | x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => {
            Some(self.rhs(ctx))
        }
        _ => None?,
    },
    // x | 0 -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => {
            Some(self.lhs(ctx))
        }
        _ => None?,
    },
    // x | x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("bitwise.xor", BitwiseXorOp);
const_eval!(BitwiseXorOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs ^ rhs,
    // x ^ x -> 0
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(int_attr(ctx, self.result_type(ctx), 0))
        } else {
            None
        }
    }
});
simplify!(BitwiseXorOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => {
            Some(self.rhs(ctx))
        }
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => {
            Some(self.lhs(ctx))
        }
        _ => None?,
    }
});

pure_binop!("bitwise.shl", ShiftLeftOp);
const_eval!(ShiftLeftOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs << rhs,
    // 0 << x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Int(0) | ConstantValue::UInt(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x << width -> 0
    custom: |_, rhs| {
        let ty = self.lhs(ctx).get_type(ctx);
        let width = width(ctx, ty);
        Some(match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if val > 0 && width <= val as usize => int_attr(ctx, ty, 0),
            ConstantValue::UInt(val) if width <= val as usize => int_attr(ctx, ty, 0),
            _ => None?
        })
    }
});
simplify!(ShiftLeftOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("bitwise.shr", ShiftRightOp);
const_eval!(ShiftRightOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs >> rhs,
    // 0 >> x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Int(0) | ConstantValue::UInt(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x >> width -> 0
    custom: |_, rhs| {
        let ty = self.lhs(ctx).get_type(ctx);
        let width = width(ctx, ty);
        Some(match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if val > 0 && width <= val as usize => int_attr(ctx, ty, 0),
            ConstantValue::UInt(val) if width <= val as usize => int_attr(ctx, ty, 0),
            _ => None?
        })
    }
});
simplify!(ShiftRightOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_unop!("bitwise.not", BitwiseNotOp);
const_eval!(BitwiseNotOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| !inp
});

macro_rules! pure_unop_u32 {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(from_inputs = |ctx, input| u32_maybe_vec(ctx, input))]
        #[$crate::prelude::op_interfaces(SameOperandsType, $crate::interfaces::TriviallyUnrollable)]
        #[$crate::prelude::op_traits($crate::CanMaterialize, $crate::Pure)]
        pub struct $ty {
            pub input: Value,
        }
    };
}

fn u32_maybe_vec(ctx: &Context, value: &Value) -> TypeHandle {
    let u32 = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
    if value.vector_size(ctx) > 1 {
        VectorType::get(ctx, u32, value.vector_size(ctx)).to_handle()
    } else {
        u32
    }
}

fn u32_ty(ctx: &Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 32, Signedness::Unsigned)
}

pure_unop_u32!("bitwise.count_ones", CountOnesOp);
const_eval!(CountOnesOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| -> IntegerAttr {
        IntegerAttr::new(u32_ty(ctx), APInt::from_u32(inp.count_ones(), bw(32)))
    },
});

pure_unop!("bitwise.reverse_bits", ReverseBitsOp);
const_eval!(ReverseBitsOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| inp.reverse_bits()
});

pure_unop_u32!("bitwise.leading_zeros", LeadingZerosBitsOp);
const_eval!(LeadingZerosBitsOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| -> IntegerAttr {
        IntegerAttr::new(u32_ty(ctx), APInt::from_u32(inp.leading_zeros(), bw(32)))
    },
});

pure_unop_u32!("bitwise.trailing_zeros", TrailingZerosBitsOp);
const_eval!(TrailingZerosBitsOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| -> IntegerAttr {
        IntegerAttr::new(u32_ty(ctx), APInt::from_u32(inp.trailing_zeros(), bw(32)))
    },
});

pure_unop_u32!("bitwise.find_first_set", FindFirstSetOp);
const_eval!(FindFirstSetOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp| -> IntegerAttr {
        let out = if inp == 0 { 0 } else { inp.trailing_zeros() + 1 };
        IntegerAttr::new(u32_ty(ctx), APInt::from_u32(out, bw(32)))
    },
});
