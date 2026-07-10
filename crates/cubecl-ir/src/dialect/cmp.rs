use cubecl_macros_internal::{const_eval, cube_op, simplify};
use half::{bf16, f16};
use pliron::{
    builtin::{attributes::IntegerAttr, types::IntegerType},
    r#type::TypeHandle,
};

use crate::{
    CanMaterialize, ConstantValue, Pure,
    attributes::{BoolAttr, FloatAttr, IndexAttr, IntAttrExt},
    dialect::{base::pure_binop, math::int_attr},
    interfaces::{TriviallyUnrollable, TypedExt},
    prelude::*,
    types::{VectorType, scalar::BoolType},
};

pure_binop!("cmp.s_min", SMinOp);
const_eval!(SMinOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs.min(rhs),
    // min(min_int, x) -> min_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    }
});
simplify!(SMinOp, {
    // min(max_int, x) -> x
    |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // min(x, max_int) -> x
    |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(self.lhs(ctx)),
            _ => None,
        }
    },
    // min(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("cmp.u_min", UMinOp);
const_eval!(UMinOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.min(rhs),
    // min(min_int, x) -> min_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(int_attr(ctx, ty, 0)),
            _ => None
        }
    }
});
simplify!(UMinOp, {
    // min(max_int, x) -> x
    |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // min(x, max_int) -> x
    |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
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

pure_binop!("cmp.f_min", FMinOp);
const_eval!(FMinOp, { [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.min(rhs) });
simplify!(FMinOp, {
    // min(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None,
    }
});

pure_binop!("cmp.s_max", SMaxOp);
const_eval!(SMaxOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs.max(rhs),
    // max(max_int, x) -> max_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    }
});
simplify!(SMaxOp, {
    // max(min_int, x) -> x
    |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // max(x, min_int) -> x
    |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(self.lhs(ctx)),
            _ => None,
        }
    },
    // max(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None,
    }
});

pure_binop!("cmp.u_max", UMaxOp);
const_eval!(UMaxOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.max(rhs),
    // max(max_int, x) -> max_int
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        let ty = const_val.get_type(ctx);
        match const_val.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    }
});
simplify!(UMaxOp, {
    // max(min_int, x) -> x
    |lhs, _| {
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(self.rhs(ctx)),
            _ => None,
        }
    },
    // max(x, min_int) -> x
    |_, rhs| {
        match rhs?.as_const_val(ctx) {
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

pure_binop!("cmp.f_max", FMaxOp);
const_eval!(FMaxOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs.max(rhs),
});
simplify!(FMaxOp, {
    // max(x, x) -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None,
    }
});

#[cube_op(name = "cmp.s_clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize)]
pub struct SClampOp {
    pub input: Value,
    pub min: Value,
    pub max: Value,
}
const_eval!(SClampOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |inp, min, max| inp.clamp(min, max),
    // clamp(x, max_int, y) -> max_int
    custom: |_, min, _| {
        let ty = min?.get_type(ctx);
        match min?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    },
    // clamp(x, y, min_int) -> min_int
    custom: |_, _, max| {
        let ty = max?.get_type(ctx);
        match max?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    }
});

#[cube_op(name = "cmp.u_clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize)]
pub struct UClampOp {
    pub input: Value,
    pub min: Value,
    pub max: Value,
}
const_eval!(UClampOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |inp, min, max| inp.clamp(min, max),
    // clamp(x, max_int, y) -> max_int
    custom: |_, min, _| {
        let ty = min?.get_type(ctx);
        match min?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(int_attr(ctx, ty, val as i128)),
            _ => None
        }
    },
    // clamp(x, y, min_int) -> min_int
    custom: |_, _, max| {
        let ty = max?.get_type(ctx);
        match max?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(int_attr(ctx, ty, 0)),
            _ => None
        }
    }
});

#[cube_op(name = "cmp.f_clamp")]
#[result_ty(same_as = input)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize)]
pub struct FClampOp {
    pub input: Value,
    pub min: Value,
    pub max: Value,
}
const_eval!(FClampOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |inp, min, max| inp.clamp(min, max),
});

macro_rules! cmp_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(from_inputs = cmp_result_ty)]
        #[$crate::prelude::op_interfaces(SameOperandsType, TriviallyUnrollable)]
        #[op_traits(Pure, CanMaterialize)]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }
    };
}

cmp_binop!("cmp.s_less_than", SLessThanOp);
const_eval!(SLessThanOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| -> BoolAttr {
        (lhs < rhs).into()
    },
    // (x < x) -> false;
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // int < min_int -> false
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // max_int < int -> false
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
});

cmp_binop!("cmp.u_less_than", ULessThanOp);
const_eval!(ULessThanOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs < rhs).into()
    },
    // (x < x) -> false;
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // int < min_int -> false
    custom: |_, rhs| {
        match rhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // max_int < int -> false
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
});

cmp_binop!("cmp.f_less_than", FLessThanOp);
const_eval!(FLessThanOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs < rhs).into()
    },
});

cmp_binop!("cmp.s_greater_than", SGreaterThanOp);
const_eval!(SGreaterThanOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| -> BoolAttr {
        (lhs > rhs).into()
    },
    // (x > x) -> false;
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // min_int > int -> false
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // int > max_int -> false
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    }
});

cmp_binop!("cmp.u_greater_than", UGreaterThanOp);
const_eval!(UGreaterThanOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs > rhs).into()
    },
    // (x > x) -> false
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    },
    // min_int > int -> false
    custom: |lhs, _| {
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(BoolAttr::new(false)),
            _ => None
        }
    },
    // int > max_int -> false
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(false)),
            _ => None
        }
    }
});

cmp_binop!("cmp.f_greater_than", FGreaterThanOp);
const_eval!(FGreaterThanOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs > rhs).into()
    },
});

cmp_binop!("cmp.s_less_than_or_equal", SLessThanOrEqualOp);
const_eval!(SLessThanOrEqualOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| -> BoolAttr {
        (lhs <= rhs).into()
    },
    // (x <= x) -> true
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // min_int <= int -> true
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // int <= max_int -> true
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.u_less_than_or_equal", ULessThanOrEqualOp);
const_eval!(ULessThanOrEqualOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs <= rhs).into()
    },
    // (x <= x) -> true
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // min_int <= int -> true
    custom: |lhs, _| {
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // int <= max_int -> true
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.f_less_than_or_equal", FLessThanOrEqualOp);
const_eval!(FLessThanOrEqualOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs <= rhs).into()
    },
});

cmp_binop!("cmp.s_greater_than_or_equal", SGreaterThanOrEqualOp);
const_eval!(SGreaterThanOrEqualOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| -> BoolAttr {
        (lhs >= rhs).into()
    },
    // (x >= x) -> true
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // int >= min_int -> true
    custom: |_, rhs| {
        let ty = rhs?.get_type(ctx);
        match rhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_min_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // max_int >= int -> true
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::Int(val) if is_max_int(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.u_greater_than_or_equal", UGreaterThanOrEqualOp);
const_eval!(UGreaterThanOrEqualOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs >= rhs).into()
    },
    // (x >= x) -> true
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    },
    // int >= min_int -> true
    custom: |_, rhs| {
        match rhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => Some(BoolAttr::new(true)),
            _ => None
        }
    },
    // max_int >= int -> true
    custom: |lhs, _| {
        let ty = lhs?.get_type(ctx);
        match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(val) if is_max_uint(ctx, ty, val) => Some(BoolAttr::new(true)),
            _ => None
        }
    }
});

cmp_binop!("cmp.f_greater_than_or_equal", FGreaterThanOrEqualOp);
const_eval!(FGreaterThanOrEqualOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs >= rhs).into()
    },
});

cmp_binop!("cmp.i_equal", IEqualOp);
const_eval!(IEqualOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64), IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs == rhs).into()
    },
    // (x == x) -> true; exclude float
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    }
});

cmp_binop!("cmp.f_equal", FEqualOp);
const_eval!(FEqualOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs == rhs).into()
    },
});

cmp_binop!("cmp.bool_equal", BoolEqualOp);
const_eval!(BoolEqualOp, {
    [BoolAttr]: |lhs, rhs| lhs == rhs,
    // (x == x) -> true; exclude float
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(true))
        } else {
            None
        }
    }
});

cmp_binop!("cmp.i_not_equal", INotEqualOp);
const_eval!(INotEqualOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64), IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| -> BoolAttr {
        (lhs != rhs).into()
    },
    // (x != x) == false; exclude float
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    }
});

cmp_binop!("cmp.f_not_equal", FNotEqualOp);
const_eval!(FNotEqualOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| -> BoolAttr {
        (lhs != rhs).into()
    },
});

cmp_binop!("cmp.bool_not_equal", BoolNotEqualOp);
const_eval!(BoolNotEqualOp, {
    [BoolAttr]: |lhs, rhs| lhs != rhs,
    // (x != x) == false; exclude float
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
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
    let ty = TypedHandle::<IntegerType>::from_handle(ty, ctx).unwrap();
    val == min_int(ty.deref(ctx).width() as usize)
}

pub(super) fn is_max_int(ctx: &Context, ty: TypeHandle, val: i64) -> bool {
    let ty = TypedHandle::<IntegerType>::from_handle(ty, ctx).unwrap();
    val == max_int(ty.deref(ctx).width() as usize)
}

pub(super) fn is_max_uint(ctx: &Context, ty: TypeHandle, val: u64) -> bool {
    val == max_uint(ty.size_bits(ctx))
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
