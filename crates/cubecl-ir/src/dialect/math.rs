use cubecl_macros_internal::cube_op;

use crate::{
    dialect::{pure_binop, pure_unop},
    interfaces::erasable,
    pliron::prelude::*,
};

use crate::interfaces::Pure;

pure_unop!("math.abs", AbsOp);
pure_unop!("math.exp", ExpOp);
pure_unop!("math.log", LogOp);
pure_unop!("math.log1p", Log1pOp);
pure_unop!("math.expm1", Expm1Op);
pure_unop!("math.sin", SinOp);
pure_unop!("math.cos", CosOp);
pure_unop!("math.tan", TanOp);
pure_unop!("math.sinh", SinhOp);
pure_unop!("math.cosh", CoshOp);
pure_unop!("math.tanh", TanhOp);
pure_unop!("math.arcsin", ArcSinOp);
pure_unop!("math.arccos", ArcCosOp);
pure_unop!("math.arctan", ArcTanOp);
pure_unop!("math.arcsinh", ArcSinhOp);
pure_unop!("math.arccosh", ArcCoshOp);
pure_unop!("math.arctanh", ArcTanhOp);
pure_unop!("math.degrees", DegreesOp);
pure_unop!("math.radians", RadiansOp);
pure_unop!("math.sqrt", SqrtOp);
pure_unop!("math.rsqrt", RsqrtOp);
pure_unop!("math.round", RoundOp);
pure_unop!("math.floor", FloorOp);
pure_unop!("math.ceil", CeilOp);
pure_unop!("math.trunc", TruncOp);
pure_unop!("math.erf", ErfOp);
pure_unop!("math.recip", RecipOp);
pure_unop!("math.neg", NegOp);
pure_unop!("math.is_nan", IsNanOp);
pure_unop!("math.is_inf", IsInfOp);

pure_binop!("math.add", AddOp);
pure_binop!("math.saturating_add", SaturatingAddOp);
pure_binop!("math.sub", SubOp);
pure_binop!("math.saturating_sub", SaturatingSubOp);
pure_binop!("math.mul", MulOp);
pure_binop!("math.div", DivOp);
pure_binop!("math.arc_tan2", ArcTan2Op);
pure_binop!("math.powf", PowfOp);
pure_binop!("math.powi", PowiOp);
pure_binop!("math.hypot", HypotOp);
pure_binop!("math.rhypot", RhypotOp);
pure_binop!("math.rem", RemOp);
pure_binop!("math.mod_floor", ModFloorOp);
pure_binop!("math.mul_hi", MulHiOp);

#[cube_op(name = "math.fma")]
#[result_ty(same_as = a)]
#[op_interfaces(SameOperandsType, SameOperandsAndResultType, Pure)]
pub struct FmaOp {
    a: Value,
    b: Value,
    c: Value,
}
erasable!(FmaOp);
