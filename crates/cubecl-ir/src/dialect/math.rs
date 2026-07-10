use core::ops::Neg;

use cubecl_macros_internal::{const_eval, cube_op, simplify};
use half::{bf16, f16};
use num::Integer;
use num_traits::Float;
use pliron::{
    attribute::AttrObj,
    builtin::{attributes::IntegerAttr, types::IntegerType},
    utils::{
        apfloat::f64_to_double,
        apint::{APInt, bw},
    },
};

use crate::{
    CanMaterialize, ConstantValue, NoMemoryEffect, NoSideEffects, Pure,
    attributes::{BoolAttr, FloatAttr, IndexAttr, IntAttrExt},
    dialect::{pure_binop, pure_unop},
    interfaces::{TriviallyUnrollable, TypedExt},
    prelude::*,
    types::{VectorType, scalar::BoolType},
};

pure_unop!("math.s_abs", SAbsOp);
const_eval!(SAbsOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |inp| inp.abs(),
});

pure_unop!("math.f_abs", FAbsOp);
const_eval!(FAbsOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |inp| inp.abs(),
});

pure_unop!("math.exp", ExpOp);
const_eval!(ExpOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.exp(),
});

pure_unop!("math.log", LogOp);
const_eval!(LogOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.ln(),
});

pure_unop!("math.log1p", Log1pOp);
const_eval!(Log1pOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.ln_1p(),
});

pure_unop!("math.expm1", Expm1Op);
const_eval!(Expm1Op, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.exp_m1(),
});

pure_unop!("math.sin", SinOp);
const_eval!(SinOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.sin(),
});

pure_unop!("math.cos", CosOp);
const_eval!(CosOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.cos(),
});

pure_unop!("math.tan", TanOp);
const_eval!(TanOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.tan(),
});

pure_unop!("math.sinh", SinhOp);
const_eval!(SinhOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.sinh(),
});

pure_unop!("math.cosh", CoshOp);
const_eval!(CoshOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.cosh(),
});

pure_unop!("math.tanh", TanhOp);
const_eval!(TanhOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.tanh(),
});

pure_unop!("math.arcsin", ArcSinOp);
const_eval!(ArcSinOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.asin(),
});

pure_unop!("math.arccos", ArcCosOp);
const_eval!(ArcCosOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.acos(),
});

pure_unop!("math.arctan", ArcTanOp);
const_eval!(ArcTanOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.atan(),
});

pure_unop!("math.arcsinh", ArcSinhOp);
const_eval!(ArcSinhOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.asinh(),
});

pure_unop!("math.arccosh", ArcCoshOp);
const_eval!(ArcCoshOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.acosh(),
});

pure_unop!("math.arctanh", ArcTanhOp);
const_eval!(ArcTanhOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.atanh(),
});

pure_unop!("math.degrees", DegreesOp);
const_eval!(DegreesOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.to_degrees(),
});

pure_unop!("math.radians", RadiansOp);
const_eval!(RadiansOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.to_radians(),
});

pure_unop!("math.sqrt", SqrtOp);
const_eval!(SqrtOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.sqrt(),
});

pure_unop!("math.rsqrt", RsqrtOp);
const_eval!(RsqrtOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.sqrt().recip(),
});

pure_unop!("math.round", RoundOp);
const_eval!(RoundOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.round(),
});

pure_unop!("math.floor", FloorOp);
const_eval!(FloorOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.floor(),
});

pure_unop!("math.ceil", CeilOp);
const_eval!(CeilOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.ceil(),
});

pure_unop!("math.trunc", TruncOp);
const_eval!(TruncOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.trunc(),
});

pure_unop!("math.erf", ErfOp);
// Unstable as of now, don't want to make a buggy version myself
// const_eval!(ErfOp, {
//     FloatAttr(f16, bf16, f32, f64): |inp| inp.erf(),
// });

pure_unop!("math.recip", RecipOp);
const_eval!(RecipOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| inp.recip(),
});

pure_unop!("math.s_neg", SNegOp);
const_eval!(SNegOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |inp| inp.neg(),
});

pure_unop!("math.f_neg", FNegOp);
const_eval!(FNegOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |inp| inp.neg(),
});

#[cube_op(name = "math.is_nan")]
#[result_ty(from_inputs = pred_result_ty)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize)]
pub struct IsNanOp {
    pub input: Value,
}
const_eval!(IsNanOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| -> BoolAttr { inp.is_nan().into() }
});

#[cube_op(name = "math.is_inf")]
#[result_ty(from_inputs = pred_result_ty)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize)]
pub struct IsInfOp {
    pub input: Value,
}
const_eval!(IsInfOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| -> BoolAttr { inp.is_infinite().into() }
});

fn pred_result_ty(ctx: &Context, input: &Value) -> TypeHandle {
    let vectorization = input.vector_size(ctx);
    let bool = BoolType::get(ctx).into();
    if vectorization == 1 {
        bool
    } else {
        VectorType::get(ctx, bool, vectorization).into()
    }
}

pure_binop!("math.i_add", IAddOp);
const_eval!(IAddOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_add(rhs),
});
simplify!(IAddOp, {
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

pure_binop!("math.f_add", FAddOp);
const_eval!(FAddOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs + rhs
});
simplify!(FAddOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Float(0.0) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Float(0.0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.saturating_s_add", SaturatingSAddOp);
const_eval!(SaturatingSAddOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs.saturating_add(rhs)
});
simplify!(SaturatingSAddOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(0) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.saturating_u_add", SaturatingUAddOp);
const_eval!(SaturatingUAddOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.saturating_add(rhs)
});
simplify!(SaturatingUAddOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::UInt(0) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::UInt(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.i_sub", ISubOp);
const_eval!(ISubOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64), IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_sub(rhs),
    // x - x -> 0
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(int_attr(ctx, self.get_type(ctx), 0))
        } else {
            None
        }
    }
});
simplify!(ISubOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) | ConstantValue::UInt(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.f_sub", FSubOp);
const_eval!(FSubOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs - rhs,
    // x - x -> 0
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(float_attr(self.get_type(ctx), 0.0))
        } else {
            None
        }
    }
});
simplify!(FSubOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Float(0.0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.saturating_s_sub", SaturatingSSubOp);
const_eval!(SaturatingSSubOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs.saturating_sub(rhs)
});
simplify!(SaturatingSSubOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.saturating_u_sub", SaturatingUSubOp);
const_eval!(SaturatingUSubOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.saturating_sub(rhs)
});
simplify!(SaturatingUSubOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::UInt(0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.i_mul", IMulOp);
const_eval!(IMulOp, {
    [IndexAttr, IntegerAttr(i8, i16, i32, i64), IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_mul(rhs),
    // 0 * x -> 0; x * 0 -> 0
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        Some(match const_val.as_const_val(ctx) {
            ConstantValue::Int(0) | ConstantValue::UInt(0) => int_attr(ctx, const_val.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(IMulOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(1) | ConstantValue::UInt(1) => {
            Some(self.rhs(ctx))
        }
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(1) | ConstantValue::UInt(1) => {
            Some(self.lhs(ctx))
        }
        _ => None?,
    }
});

pure_binop!("math.f_mul", FMulOp);
const_eval!(FMulOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs * rhs,
    // 0 * x -> 0; x * 0 -> 0
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        Some(match const_val.as_const_val(ctx) {
            ConstantValue::Float(0.0) => float_attr( const_val.get_type(ctx), 0.0),
            _ => None?
        })
    }
});
simplify!(FMulOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Float(1.0) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Float(1.0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

#[cube_op(name = "math.s_div")]
#[result_ty(same_as = lhs)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoSideEffects, NoMemoryEffect)] // Not pure because divide by zero
pub struct SDivOp {
    pub lhs: Value,
    pub rhs: Value,
}
const_eval!(SDivOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs.wrapping_div(rhs),
    // 0 / x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Int(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x / x -> 1
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    }
});
simplify!(SDivOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

#[cube_op(name = "math.u_div")]
#[result_ty(same_as = lhs)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoSideEffects, NoMemoryEffect)] // Not pure because divide by zero
pub struct UDivOp {
    pub lhs: Value,
    pub rhs: Value,
}
const_eval!(UDivOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_div(rhs),
    // 0 / x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x / x -> 1
    custom: |_, _| {
        if self.lhs(ctx) == self.rhs(ctx) {
            Some(BoolAttr::new(false))
        } else {
            None
        }
    }
});
simplify!(UDivOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::UInt(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.f_div", FDivOp);
const_eval!(FDivOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs / rhs,
    // 0 / x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Float(0.0) => float_attr(lhs?.get_type(ctx), 0.0),
            _ => None?
        })
    },
});
simplify!(FDivOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Float(1.0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.arc_tan2", ArcTan2Op);
const_eval!(ArcTan2Op, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.atan2(rhs),
});

pure_binop!("math.powf", PowfOp);
const_eval!(PowfOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.powf(rhs),
});

#[cube_op(name = "math.powi")]
#[result_ty(same_as = lhs)]
#[op_traits(Pure, CanMaterialize)]
pub struct PowiOp {
    pub lhs: Value,
    pub rhs: Value,
}

// TODO const_eval

pure_binop!("math.hypot", HypotOp);
const_eval!(HypotOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.hypot(rhs),
});

pure_binop!("math.rhypot", RhypotOp);
const_eval!(RhypotOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.hypot(rhs).recip(),
});

#[cube_op(name = "math.s_rem")]
#[result_ty(same_as = lhs)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoSideEffects, NoMemoryEffect)] // Not pure because divide by zero
pub struct SRemOp {
    pub lhs: Value,
    pub rhs: Value,
}
const_eval!(SRemOp, {
    [IntegerAttr(i8, i16, i32, i64)]: |lhs, rhs| lhs % rhs,
    // 0 % x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Int(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x % 1 -> 0
    custom: |_, rhs| {
        Some(match rhs?.as_const_val(ctx) {
            ConstantValue::Int(1) => int_attr(ctx, rhs?.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(SRemOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

#[cube_op(name = "math.u_rem")]
#[result_ty(same_as = lhs)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoSideEffects, NoMemoryEffect)] // Not pure because divide by zero
pub struct URemOp {
    pub lhs: Value,
    pub rhs: Value,
}
const_eval!(URemOp, {
    [IndexAttr, IntegerAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs % rhs,
    // 0 % x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::UInt(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x % 1 -> 0
    custom: |_, rhs| {
        Some(match rhs?.as_const_val(ctx) {
            ConstantValue::UInt(1) => int_attr(ctx, rhs?.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(URemOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::UInt(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.f_rem", FRemOp);
const_eval!(FRemOp, {
    [FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs % rhs,
    // 0 % x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Float(0.0) => float_attr( lhs?.get_type(ctx), 0.0),
            _ => None?
        })
    },
});
simplify!(FRemOp, {
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Float(1.0) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

#[cube_op(name = "math.s_mod_floor")]
#[result_ty(same_as = lhs)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoSideEffects, NoMemoryEffect)] // Not pure because divide by zero
pub struct SModFloorOp {
    pub lhs: Value,
    pub rhs: Value,
}
const_eval!(SModFloorOp, {
    IntegerAttr(i8, i16, i32, i64): |lhs, rhs| lhs.mod_floor(&rhs),
    // 0 % x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Int(0) => int_attr(ctx, lhs?.get_type(ctx), 0),
            _ => None?
        })
    },
    // x % 1 -> 0
    custom: |_, rhs| {
        Some(match rhs?.as_const_val(ctx) {
            ConstantValue::Int(1) => int_attr(ctx, rhs?.get_type(ctx), 0),
            _ => None?
        })
    }
});

pure_binop!("math.f_mod_floor", FModFloorOp);
const_eval!(FModFloorOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs - (lhs / rhs).floor() * rhs,
    // 0 % x -> 0
    custom: |lhs, _| {
        Some(match lhs?.as_const_val(ctx) {
            ConstantValue::Float(0.0) => float_attr( lhs?.get_type(ctx), 0.0),
            _ => None?
        })
    },
});

pure_binop!("math.s_mul_hi", SMulHiOp);
const_eval!(SMulHiOp, {
    IntegerAttr(i64): |lhs, rhs| ((lhs as i128 * rhs as i128) >> 64) as i64,
    IntegerAttr(i32): |lhs, rhs| ((lhs as i64 * rhs as i64) >> 32) as i32,
    // 0 * x -> 0; x * 0 -> 0
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        Some(match const_val.as_const_val(ctx) {
            ConstantValue::Int(0) => int_attr(ctx, const_val.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(SMulHiOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Int(1) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Int(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pure_binop!("math.u_mul_hi", UMulHiOp);
const_eval!(UMulHiOp, {
    IndexAttr: |lhs, rhs| ((lhs as u128 * rhs as u128) >> 64) as usize,
    IntegerAttr(u64): |lhs, rhs| ((lhs as u128 * rhs as u128) >> 64) as u64,
    IntegerAttr(u32): |lhs, rhs| ((lhs as u64 * rhs as u64) >> 32) as u32,
    // 0 * x -> 0; x * 0 -> 0
    custom: |lhs, rhs| {
        let const_val = lhs.or(rhs)?;
        Some(match const_val.as_const_val(ctx) {
            ConstantValue::UInt(0) => int_attr(ctx, const_val.get_type(ctx), 0),
            _ => None?
        })
    }
});
simplify!(UMulHiOp, {
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::UInt(1) => Some(self.rhs(ctx)),
        _ => None?,
    },
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::UInt(1) => Some(self.lhs(ctx)),
        _ => None?,
    }
});

pub(super) fn index_attr(val: usize) -> AttrObj {
    AttrObj::from(IndexAttr::new(val))
}

pub(super) fn int_attr(ctx: &Context, ty: TypeHandle, val: i128) -> AttrObj {
    if ty.is_index(ctx) {
        IndexAttr::new(val as usize).into()
    } else {
        let ty = TypedHandle::<IntegerType>::from_handle(ty, ctx).unwrap();
        let width = bw(ty.deref(ctx).width() as usize);
        let val = APInt::from_i128(val, width);
        AttrObj::from(IntegerAttr::new(ty, val))
    }
}

pub(super) fn float_attr(ty: TypeHandle, val: f64) -> AttrObj {
    AttrObj::from(FloatAttr::new(ty, f64_to_double(val)))
}

#[cube_op(name = "math.fma")]
#[result_ty(same_as = a)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType)]
#[op_traits(Pure, CanMaterialize)]
pub struct FmaOp {
    pub a: Value,
    pub b: Value,
    pub c: Value,
}
const_eval!(FmaOp, {
    FloatAttr(f16, bf16, f32, f64): |a, b, c| a * b + c,
});
