use core::ops::Neg;

use cubecl_macros_internal::{const_eval, cube_op};
use half::{bf16, f16};
use num::Integer;
use num_traits::Float;

use crate::{
    attributes::{BoolAttr, FloatAttr, IndexAttr, IntAttr, UIntAttr},
    dialect::{pure_binop, pure_unop},
    interfaces::erasable,
    prelude::*,
};

use crate::interfaces::Pure;

pure_unop!("math.abs", AbsOp);
const_eval!(AbsOp, {
    [IntAttr(i8, i16, i32, i64), FloatAttr(f16, bf16, f32, f64)]: |inp| inp.abs(),
    [IndexAttr, UIntAttr(u8, u16, u32, u64)]: |inp| inp
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

pure_unop!("math.neg", NegOp);
const_eval!(NegOp, {
    [IntAttr(i8, i16, i32, i64), FloatAttr(f16, bf16, f32, f64)]: |inp| inp.neg(),
});

pure_unop!("math.is_nan", IsNanOp);
const_eval!(IsNanOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| -> BoolAttr { inp.is_nan().into() }
});

pure_unop!("math.is_inf", IsInfOp);
const_eval!(IsInfOp, {
    FloatAttr(f16, bf16, f32, f64): |inp| -> BoolAttr { inp.is_infinite().into() }
});

pure_binop!("math.add", AddOp);
const_eval!(AddOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_add(rhs),
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs + rhs
});

pure_binop!("math.saturating_add", SaturatingAddOp);
const_eval!(SaturatingAddOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.saturating_add(rhs)
});

pure_binop!("math.sub", SubOp);
const_eval!(SubOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_sub(rhs),
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs - rhs
});

pure_binop!("math.saturating_sub", SaturatingSubOp);
const_eval!(SaturatingSubOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.saturating_sub(rhs)
});

pure_binop!("math.mul", MulOp);
const_eval!(MulOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_mul(rhs),
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs * rhs
});

pure_binop!("math.div", DivOp);
const_eval!(DivOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs.wrapping_div(rhs),
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs / rhs
});

pure_binop!("math.arc_tan2", ArcTan2Op);
const_eval!(ArcTan2Op, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.atan2(rhs),
});

pure_binop!("math.powf", PowfOp);
const_eval!(PowfOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.powf(rhs),
});

pure_binop!("math.powi", PowiOp);
// TODO const_eval

pure_binop!("math.hypot", HypotOp);
const_eval!(HypotOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.hypot(rhs),
});

pure_binop!("math.rhypot", RhypotOp);
const_eval!(RhypotOp, {
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs.hypot(rhs).recip(),
});

pure_binop!("math.rem", RemOp);
const_eval!(RemOp, {
    [IndexAttr, IntAttr(i8, i16, i32, i64), UIntAttr(u8, u16, u32, u64), FloatAttr(f16, bf16, f32, f64)]: |lhs, rhs| lhs % rhs,
});

pure_binop!("math.mod_floor", ModFloorOp);
const_eval!(ModFloorOp, {
    [IndexAttr, UIntAttr(u8, u16, u32, u64)]: |lhs, rhs| lhs % rhs,
    IntAttr(i8, i16, i32, i64): |lhs, rhs| lhs.mod_floor(&rhs),
    FloatAttr(f16, bf16, f32, f64): |lhs, rhs| lhs - (lhs / rhs).floor() * rhs
});

pure_binop!("math.mul_hi", MulHiOp);
const_eval!(MulHiOp, {
    IndexAttr: |lhs, rhs| ((lhs as u128 * rhs as u128) >> 64) as usize,
    UIntAttr(u64): |lhs, rhs| ((lhs as u128 * rhs as u128) >> 64) as u64,
    IntAttr(i64): |lhs, rhs| ((lhs as i128 * rhs as i128) >> 64) as i64,
    UIntAttr(u32): |lhs, rhs| ((lhs as u64 * rhs as u64) >> 32) as u32,
    IntAttr(i32): |lhs, rhs| ((lhs as i64 * rhs as i64) >> 32) as i32,
});

#[cube_op(name = "math.fma")]
#[result_ty(same_as = a)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct FmaOp {
    pub a: Value,
    pub b: Value,
    pub c: Value,
}
erasable!(FmaOp);
const_eval!(FmaOp, {
    FloatAttr(f16, bf16, f32, f64): |a, b, c| a * b + c,
});
