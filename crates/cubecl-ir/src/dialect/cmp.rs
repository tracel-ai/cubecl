use cubecl_macros_internal::{const_eval, cube_op, simplify};
use half::{bf16, f16};
use pliron::r#type::TypeHandle;

use crate::{
    ConstantValue,
    attributes::{BoolAttr, FloatAttr, IndexAttr, IntAttr, UIntAttr},
    dialect::{
        base::pure_binop,
        math::{int_attr, uint_attr},
    },
    interfaces::{Pure, TypedExt, erasable},
    prelude::*,
    types::{
        VectorType,
        scalar::{BoolType, IntType},
    },
};

pure_binop!("cmp.min", MinOp);
const_eval!(MinOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.min(rhs),
    // min(min_int, x) -> min_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val() {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(int_attr(ctx, ty, val)),
            ConstantValue::UInt(0) => Some(uint_attr(ctx, ty, 0)),
            _ => None
        }
    }
});
simplify!(MinOp, {
    // min(max_int, x) -> x
    |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(self.rhs(ctx)),
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // min(x, max_int) -> x
    |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(self.lhs(ctx)),
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(self.lhs(ctx)),
            _ => None,
        }
    },
    // min(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("cmp.max", MaxOp);
const_eval!(MaxOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.max(rhs),
    // max(max_int, x) -> max_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val() {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(int_attr(ctx, ty, val)),
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(uint_attr(ctx, ty, val)),
            _ => None
        }
    }
});
simplify!(MaxOp, {
    // max(min_int, x) -> x
    |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(self.rhs(ctx)),
            ConstantValue::UInt(0) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // max(x, min_int) -> x
    |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(self.lhs(ctx)),
            ConstantValue::UInt(0) => Some(self.lhs(ctx)),
            _ => None,
        }
    },
    // max(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None,
    }
});

#[cube_op(name = "cmp.clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct ClampOp {
    pub input: Value,
    pub min: Value,
    pub max: Value,
}
erasable!(ClampOp);
const_eval!(ClampOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |inp, min, max| inp.clamp(min, max),
    // clamp(x, max_int, y) -> max_int
    custom: |_, min, _| {
        let ty = min?.get_type(ctx);
        match min?.as_const_val() {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(int_attr(ctx, ty, val)),
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(uint_attr(ctx, ty, val)),
            _ => None
        }
    },
    // clamp(x, y, min_int) -> min_int
    custom: |_, _, max| {
        let ty = max?.get_type(ctx);
        match max?.as_const_val() {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(int_attr(ctx, ty, val)),
            ConstantValue::UInt(0) => Some(uint_attr(ctx, ty, 0)),
            _ => None
        }
    }
});

macro_rules! cmp_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(from_inputs = cmp_result_ty)]
        #[$crate::prelude::op_interfaces(SameOperandsType, Pure)]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }

        $crate::interfaces::erasable!($ty);
    };
}

cmp_binop!("cmp.less_than", LessThanOp);
const_eval!(LessThanOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs < rhs).into()
    },
    // (x < x) -> false; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // int < min_int -> false
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::UInt(0) => Some(BoolAttr::new(false)),
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // max_int < int -> false
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(false)),
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
});

cmp_binop!("cmp.greater_than", GreaterThanOp);
const_eval!(GreaterThanOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs > rhs).into()
    },
    // (x > x) -> false; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // min_int > int -> false
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::UInt(0) => Some(BoolAttr::new(false)),
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // int > max_int -> false
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(false)),
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    }
});

cmp_binop!("cmp.less_than_or_equal", LessThanOrEqualOp);
const_eval!(LessThanOrEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs <= rhs).into()
    },
    // (x <= x) -> true; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // min_int <= int -> true
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::UInt(0) => Some(BoolAttr::new(true)),
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // int <= max_int -> true
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(true)),
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.greater_than_or_equal", GreaterThanOrEqualOp);
const_eval!(GreaterThanOrEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs >= rhs).into()
    },
    // (x >= x) -> true; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // int >= min_int -> true
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val() {
            ConstantValue::UInt(0) => Some(BoolAttr::new(true)),
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // max_int >= int -> true
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val() {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(true)),
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.equal", EqualOp);
const_eval!(EqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs == rhs).into()
    },
    // (x == x) -> true; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    }
});

cmp_binop!("cmp.not_equal", NotEqualOp);
const_eval!(NotEqualOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]:
    |lhs, rhs| -> BoolAttr {
        (lhs != rhs).into()
    },
    // (x != x) == false; exclude float
    custom: |_, _| {
        let lhs = self.lhs(ctx);
        if lhs == self.rhs(ctx) && (lhs.is_int(ctx) || lhs.is_uint(ctx) || lhs.is_bool(ctx)) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    }
});

fn cmp_result_ty(ctx: &Context, lhs: &Value, _: &Value) -> TypeHandle {
    let vectorization = lhs.vector_size(ctx);
    let bool = BoolType::get(ctx).into();
    if vectorization == 1 {
        bool
    } else {
        VectorType::get(ctx, bool, vectorization).into()
    }
}

pub(super) fn width(ctx: &Context, ty: TypeHandle) -> usize {
    ty.size(ctx) * 8
}

fn is_min_int(ctx: &Context, ty: TypeHandle, val: i64) -> bool {
    let ty = TypedHandle::<IntType>::from_handle(ty, ctx).unwrap();
    val == min_int(ty.deref(ctx).width)
}

pub(super) fn is_max_int(ctx: &Context, ty: TypeHandle, val: i64) -> bool {
    let ty = TypedHandle::<IntType>::from_handle(ty, ctx).unwrap();
    val == max_int(ty.deref(ctx).width)
}

pub(super) fn is_max_uint(ctx: &Context, ty: TypeHandle, val: u64) -> bool {
    let ty = TypedHandle::<IntType>::from_handle(ty, ctx).unwrap();
    val == max_uint(ty.deref(ctx).width)
}

fn min_int(width: usize) -> i64 {
    if width >= 64 {
        i64::MIN
    } else {
        -(1i64 << (width - 1))
    }
}

fn max_int(width: usize) -> i64 {
    if width >= 64 {
        i64::MAX
    } else {
        (1i64 << (width - 1)) - 1
    }
}

fn max_uint(width: usize) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}
