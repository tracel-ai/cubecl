use cubecl_core::{
    self as cubecl, WgpuCompilationOptions,
    prelude::{polyfills::*, *},
};
use cubecl_ir::{Scope, dialect::math::*, interfaces::TypedExt, prelude::*, try_cast_ty};
use pliron::builtin::types::{IntegerType, Signedness};

use crate::compiler::wgsl::{
    lower::{LowerOp, lower_binop, lower_unop},
    to_wgsl::wgsl_op_with_out,
};

wgsl_op_with_out!(SAbsOp, |op, ctx| {
    format!("abs({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(FAbsOp, |op, ctx| {
    format!("abs({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(ExpOp, |op, ctx| {
    format!("exp({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(LogOp, |op, ctx| {
    format!("log({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SinOp, |op, ctx| {
    format!("sin({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(CosOp, |op, ctx| {
    format!("cos({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(TanOp, |op, ctx| {
    format!("tan({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SinhOp, |op, ctx| {
    format!("sinh({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(CoshOp, |op, ctx| {
    format!("cosh({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(TanhOp, |op, ctx| {
    format!("tanh({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(ArcSinOp, |op, ctx| {
    format!("asin({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ArcCosOp, |op, ctx| {
    format!("acos({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ArcTanOp, |op, ctx| {
    format!("atan({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(ArcSinhOp, |op, ctx| {
    format!("asinh({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ArcCoshOp, |op, ctx| {
    format!("acosh({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ArcTanhOp, |op, ctx| {
    format!("atanh({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(ArcTan2Op, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    format!("atan2({lhs}, {})", op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(DegreesOp, |op, ctx| {
    format!("degrees({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(RadiansOp, |op, ctx| {
    format!("radians({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(RoundOp, |op, ctx| {
    format!("round({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(FloorOp, |op, ctx| {
    format!("floor({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(CeilOp, |op, ctx| {
    format!("ceil({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(TruncOp, |op, ctx| {
    format!("trunc({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SqrtOp, |op, ctx| {
    format!("sqrt({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(RsqrtOp, |op, ctx| {
    format!("inverseSqrt({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SNegOp, |op, ctx| {
    format!("-{}", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(FNegOp, |op, ctx| {
    format!("-{}", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(IAddOp, |op, ctx| {
    format!("{} + {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FAddOp, |op, ctx| {
    format!("{} + {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(ISubOp, |op, ctx| {
    format!("{} - {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FSubOp, |op, ctx| {
    format!("{} - {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(IMulOp, |op, ctx| {
    format!("{} * {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FMulOp, |op, ctx| {
    format!("{} * {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SDivOp, |op, ctx| {
    format!("{} / {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UDivOp, |op, ctx| {
    format!("{} / {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FDivOp, |op, ctx| {
    format!("{} / {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SRemOp, |op, ctx| {
    format!("{} % {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(URemOp, |op, ctx| {
    format!("{} % {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FRemOp, |op, ctx| {
    format!("{} % {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(FmaOp, |op, ctx| {
    let a = op.a(ctx).name(ctx);
    let b = op.b(ctx).name(ctx);
    let c = op.c(ctx).name(ctx);
    format!("fma({a}, {b}, {c})",)
});

wgsl_op_with_out!(SimplePowOp, |op, ctx| {
    format!("pow({}, {})", op.base(ctx).name(ctx), op.exp(ctx).name(ctx))
});

#[cube_op(name = "wgsl.tanh")]
#[result_ty(same_as = input)]
pub struct WgslTanhOp {
    pub input: Value,
}

wgsl_op_with_out!(WgslTanhOp, |op, ctx| {
    format!("tanh({})", op.input(ctx).name(ctx))
});

lower_unop!(ErfOp, erf);
lower_unop!(Log1pOp, log1p);
lower_unop!(Expm1Op, expm1);
lower_unop!(RecipOp, recip);
lower_unop!(IsNanOp, is_nan);
lower_unop!(IsInfOp, is_inf);

lower_binop!(PowfOp, powf);
lower_binop!(PowiOp, powi);
lower_binop!(HypotOp, hypot);
lower_binop!(RhypotOp, rhypot);
lower_binop!(SModFloorOp, s_mod_floor);
lower_binop!(FModFloorOp, f_mod_floor);

lower_unop!(TanhOp, safe_tanh, |_, _| cfg!(target_os = "macos"));

/// use the simple version because otherwise we'd get an infinite lowering loop
#[cube]
fn simple_tanh<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let input = input.read_value(scope);
        let tanh = WgslTanhOp::new(scope.ctx_mut(), input);
        scope.register_with_result(&tanh).into()
    })
}

#[cube]
fn safe_tanh<T: Float, N: Size>(x: Vector<T, N>) -> Vector<T, N> {
    let threshold = Vector::new(T::new(43.0));
    select(x > threshold, Vector::one(), simple_tanh(x))
}

define_scalar!(UInt);

#[cube]
#[allow(clippy::extra_unused_type_parameters)]
fn select_inf_bits_abs_mask_uint<F: Float>() -> comptime_type!((i64, i64)) {
    intrinsic!(|scope| {
        let width = F::__expand_size_bits(scope) as u32;
        let uint = IntegerType::get(scope.ctx(), width, Signedness::Unsigned);
        scope.register_value_type::<UInt, ()>(uint.to_handle());

        let ty = F::__expand_as_type(scope).deref(scope.ctx());
        let semantics = try_cast_ty!(ty, scope.ctx(), dyn FloatTypeInterface).get_semantics();
        let inf_bits = ((1u128 << semantics.exp_bits) - 1) << (semantics.precision - 1);
        let abs_mask = inf_bits | semantics.nan_payload_mask | semantics.nan_significand_base;
        (inf_bits as i64, abs_mask as i64)
    })
}

#[cube]
fn is_nan<F: Float, N: Size>(x: Vector<F, N>) -> Vector<bool, N> {
    let (abs_bits, inf_bits) = abs_and_inf_bits::<F, N>(x);
    abs_bits.greater_than(&inf_bits)
}

#[cube]
fn is_inf<F: Float, N: Size>(x: Vector<F, N>) -> Vector<bool, N> {
    let (abs_bits, inf_bits) = abs_and_inf_bits::<F, N>(x);
    abs_bits.equal(&inf_bits)
}

#[cube]
fn abs_and_inf_bits<F: Float, N: Size>(x: Vector<F, N>) -> (Vector<UInt, N>, Vector<UInt, N>) {
    let size = F::size().comptime();
    // WGSL doesn't support u16, so for f16 we convert to f32 first (NaN/Inf are preserved)
    let (bits, inf_bits, abs_mask) = if size < 4 {
        let x = Vector::<f32, N>::cast_from(x);
        let (inf_bits, abs_mask) = select_inf_bits_abs_mask_uint::<f32>();
        let bits = Vector::<UInt, N>::reinterpret(x);
        (bits, inf_bits, abs_mask)
    } else {
        let (inf_bits, abs_mask) = select_inf_bits_abs_mask_uint::<F>();
        let bits = Vector::<UInt, N>::reinterpret(x);
        (bits, inf_bits, abs_mask)
    };
    let inf_bits = Vector::new(UInt::from_int(inf_bits));
    let abs_bits = bits & Vector::new(UInt::from_int(abs_mask));
    (abs_bits, inf_bits)
}

#[cube]
fn s_mod_floor<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let floored = (Vector::<f32, N>::cast_from(lhs) / Vector::<f32, N>::cast_from(rhs)).floor();
    lhs - rhs * Vector::cast_from(floored)
}

#[cube]
fn f_mod_floor<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    lhs - rhs * (lhs / rhs).floor()
}

#[op_interface_impl]
impl LowerOp for SMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let has_u64 = ctx.aux_ty::<WgpuCompilationOptions>().supports_u64;
        let lhs = self.lhs(ctx);
        let val = if lhs.is_int_of_width(ctx, 32) && has_u64 {
            expand_s_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}

#[op_interface_impl]
impl LowerOp for UMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let has_u64 = ctx.aux_ty::<WgpuCompilationOptions>().supports_u64;
        let lhs = self.lhs(ctx);
        let val = if lhs.is_int_of_width(ctx, 32) && has_u64 {
            expand_u_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}
