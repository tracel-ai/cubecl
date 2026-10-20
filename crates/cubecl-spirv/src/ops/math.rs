use cubecl_core::{self as cubecl, frontend::polyfills::*, prelude::*};
use cubecl_ir::{dialect::math, interfaces::TypedExt, prelude::*};
use cubecl_opt::passes::uniformity::op_dyn_uniformity;
use pliron_spirv::{
    attrs::PackedVectorFormatAttr, ext::gl, ops, spirv::PackedVectorFormat, types::StructType,
};

use crate::{
    decorate_uniform,
    lower::{LowerOp, lower_binop, lower_unop},
    ops::{
        base::{binop_to_spirv_dialect, ternop_to_spirv_dialect, unop_to_spirv_dialect},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

unop_to_spirv_dialect!(math::SNegOp => ops::SNegateOp);
unop_to_spirv_dialect!(math::FNegOp => ops::FNegateOp);

binop_to_spirv_dialect!(math::IAddOp => ops::IAddOp);
binop_to_spirv_dialect!(math::FAddOp => ops::FAddOp);
binop_to_spirv_dialect!(math::ISubOp => ops::ISubOp);
binop_to_spirv_dialect!(math::FSubOp => ops::FSubOp);
binop_to_spirv_dialect!(math::IMulOp => ops::IMulOp);
binop_to_spirv_dialect!(math::FMulOp => ops::FMulOp);
binop_to_spirv_dialect!(math::SDivOp => ops::SDivOp);
binop_to_spirv_dialect!(math::UDivOp => ops::UDivOp);
binop_to_spirv_dialect!(math::FDivOp => ops::FDivOp);
lower_binop!(math::SRemOp, signed_remainder);
binop_to_spirv_dialect!(math::URemOp => ops::UModOp);
binop_to_spirv_dialect!(math::FRemOp => ops::FRemOp);
lower_binop!(math::SModFloorOp, signed_mod_floor);
binop_to_spirv_dialect!(math::FModFloorOp => ops::FModOp);

// Vulkan only defines OpSRem/OpSMod for negative operands when maintenance8 is
// enabled. Reconstruct the truncating remainder using integer arithmetic so
// signed inputs work on devices without that feature too.
#[cube]
fn signed_remainder<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    // Division by zero and MIN / -1 are outside the portable contract.
    lhs - (lhs / rhs) * rhs
}

#[cube]
fn signed_mod_floor<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let remainder = signed_remainder(lhs, rhs);
    let zero = Vector::new(I::from_int(0));
    let different_signs = (remainder ^ rhs).less_than(&zero);
    let adjust = remainder.not_equal(&zero).vec_and(different_signs);
    remainder + select_many(adjust, rhs, zero)
}

#[op_interface_impl]
impl LowerOp for math::Dp4aOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        !ctx.aux_ty::<cubecl_core::WgpuCompilationOptions>()
            .vulkan
            .supports_dp4a
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        vec![expand_dp4a_polyfill(
            scope,
            self.a(ctx),
            self.b(ctx),
            self.c(ctx),
        )]
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for math::Dp4aOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let packed = PackedVectorFormatAttr::new(PackedVectorFormat::PackedVectorFormat4x8Bit);
        let dot = ops::SDotOp::new(ctx, out_ty, self.a(ctx), self.b(ctx), Some(packed));
        rewriter.append_op(ctx, &dot);
        let add = ops::IAddOp::new(ctx, out_ty, dot.get_result(ctx), self.c(ctx));
        rewriter.append_op(ctx, &add);
        rewriter.replace_operation(ctx, op, add.get_operation());
        Ok(())
    }
}

binop_to_spirv_dialect!(SimplePowOp => gl::PowOp);

unop_to_spirv_dialect!(math::SAbsOp => gl::SAbsOp);
unop_to_spirv_dialect!(math::FAbsOp => gl::FAbsOp);
unop_to_spirv_dialect!(math::ExpOp => gl::ExpOp);
unop_to_spirv_dialect!(math::LogOp => gl::LogOp);
unop_to_spirv_dialect!(math::IsNanOp => ops::IsNanOp);
unop_to_spirv_dialect!(math::IsInfOp => ops::IsInfOp);

unop_to_spirv_dialect!(math::SinOp => gl::SinOp);
unop_to_spirv_dialect!(math::CosOp => gl::CosOp);
unop_to_spirv_dialect!(math::TanOp => gl::TanOp);
unop_to_spirv_dialect!(math::SinhOp => gl::SinhOp);
unop_to_spirv_dialect!(math::CoshOp => gl::CoshOp);
unop_to_spirv_dialect!(math::TanhOp => gl::TanhOp);
unop_to_spirv_dialect!(math::ArcSinOp => gl::AsinOp);
unop_to_spirv_dialect!(math::ArcCosOp => gl::AcosOp);
unop_to_spirv_dialect!(math::ArcTanOp => gl::AtanOp);
unop_to_spirv_dialect!(math::ArcSinhOp => gl::AsinhOp);
unop_to_spirv_dialect!(math::ArcCoshOp => gl::AcoshOp);
unop_to_spirv_dialect!(math::ArcTanhOp => gl::AtanhOp);
binop_to_spirv_dialect!(math::ArcTan2Op => gl::Atan2Op);

unop_to_spirv_dialect!(math::DegreesOp => gl::DegreesOp);
unop_to_spirv_dialect!(math::RadiansOp => gl::RadiansOp);

// GLSL.std.450 defines its trigonometric, exponential and power instructions only for 16- and
// 32-bit floats. A driver can accept them on f64 and return wrong values, so an f64 operation
// runs in f32 and widens the result: it has f32 precision and range.
macro_rules! f64_through_f32 {
    ($ty: ty, $($operand: ident),+) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                self.get_result(ctx).scalar_ty(ctx).is_float64(ctx)
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let ctx = scope.ctx();
                let uniformity = op_dyn_uniformity(ctx, self.get_operation());
                $(let $operand = self.$operand(ctx);)+
                scope.register_value_type::<T, S>(self.get_result(ctx));
                $(let $operand = narrow_f64::expand::<T, S>(scope, $operand.into()).read_value(scope);)+
                let op = <$ty>::new(scope.ctx_mut(), $($operand),+);
                decorate_uniform(scope.ctx(), op.get_operation(), uniformity);
                let value = scope.register_with_result(&op);
                vec![widen_f64::expand::<T, S>(scope, value.into()).read_value(scope)]
            }
        }
    };
}

f64_through_f32!(math::SinOp, input);
f64_through_f32!(math::CosOp, input);
f64_through_f32!(math::TanOp, input);
f64_through_f32!(math::SinhOp, input);
f64_through_f32!(math::CoshOp, input);
f64_through_f32!(math::TanhOp, input);
f64_through_f32!(math::ArcSinOp, input);
f64_through_f32!(math::ArcCosOp, input);
f64_through_f32!(math::ArcTanOp, input);
f64_through_f32!(math::ArcSinhOp, input);
f64_through_f32!(math::ArcCoshOp, input);
f64_through_f32!(math::ArcTanhOp, input);
f64_through_f32!(math::ArcTan2Op, lhs, rhs);
f64_through_f32!(math::ExpOp, input);
f64_through_f32!(math::LogOp, input);
f64_through_f32!(SimplePowOp, base, exp);

#[cube]
fn narrow_f64<F: Float, N: Size>(x: Vector<F, N>) -> Vector<f32, N> {
    Vector::cast_from(x)
}

#[cube]
fn widen_f64<F: Float, N: Size>(x: Vector<f32, N>) -> Vector<F, N> {
    Vector::cast_from(x)
}

// The conversion between degrees and radians is a multiplication, which f64 has, so it keeps
// f64 precision.
lower_unop!(math::DegreesOp, degrees_f64, |op, ctx| {
    op.get_result(ctx).scalar_ty(ctx).is_float64(ctx)
});
lower_unop!(math::RadiansOp, radians_f64, |op, ctx| {
    op.get_result(ctx).scalar_ty(ctx).is_float64(ctx)
});

const DEGREES_PER_RADIAN: f64 = 180.0 / core::f64::consts::PI;
const RADIANS_PER_DEGREE: f64 = core::f64::consts::PI / 180.0;

#[cube]
fn degrees_f64<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    x * Vector::new(F::new(DEGREES_PER_RADIAN))
}

#[cube]
fn radians_f64<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    x * Vector::new(F::new(RADIANS_PER_DEGREE))
}

unop_to_spirv_dialect!(math::SqrtOp => gl::SqrtOp);
unop_to_spirv_dialect!(math::RsqrtOp => gl::InverseSqrtOp);

unop_to_spirv_dialect!(math::RoundOp => gl::RoundOp);
unop_to_spirv_dialect!(math::FloorOp => gl::FloorOp);
unop_to_spirv_dialect!(math::CeilOp => gl::CeilOp);
unop_to_spirv_dialect!(math::TruncOp => gl::TruncOp);

#[op_interface_impl]
impl ToSpirvDialectOp for math::SMulHiOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let uniformity = op_dyn_uniformity(ctx, op);
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::SMulExtendedOp::new(ctx, out_st, lhs, rhs);
        decorate_uniform(ctx, mul.get_operation(), uniformity);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
        decorate_uniform(ctx, new_op.get_operation(), uniformity);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for math::UMulHiOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let uniformity = op_dyn_uniformity(ctx, op);
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::UMulExtendedOp::new(ctx, out_st, lhs, rhs);
        decorate_uniform(ctx, mul.get_operation(), uniformity);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
        decorate_uniform(ctx, new_op.get_operation(), uniformity);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for gl::PowOp {
    fn should_convert(&self, ctx: &Context) -> bool {
        ty_to_spirv_dialect(ctx, self.result_type(ctx)) != self.result_type(ctx)
    }
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        rewriter.set_value_type(ctx, self.get_result(ctx), ty);
        Ok(())
    }
}

lower_unop!(math::RecipOp, recip);
lower_unop!(math::Log1pOp, log1p);
lower_unop!(math::Expm1Op, expm1);
lower_unop!(math::CbrtOp, cbrt);
lower_unop!(math::RcbrtOp, inverse_cbrt);
lower_unop!(math::ErfOp, erf);
lower_binop!(math::HypotOp, hypot);
lower_binop!(math::RhypotOp, rhypot);

lower_binop!(math::PowfOp, powf);

#[op_interface_impl]
impl LowerOp for math::PowiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let lhs = self.lhs(scope.ctx());
        let rhs = self.rhs(scope.ctx());
        scope.register_value_type::<T, S>(lhs);

        let ty = lhs.scalar_ty(scope.ctx());
        let result = if ty.is_int(scope.ctx()) || ty.is_index(scope.ctx()) {
            powi_int::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)
        } else {
            powi::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)
        };
        vec![result]
    }
}

ternop_to_spirv_dialect!(math::FmaOp => gl::FmaOp);

/// `gl::FmaOp` is defined for floats only, so an integer fma keeps its multiply and add.
#[op_interface_impl]
impl LowerOp for math::FmaOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        self.get_result(ctx).scalar_ty(ctx).is_int(ctx)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let a = self.a(scope.ctx());
        let b = self.b(scope.ctx());
        let c = self.c(scope.ctx());
        scope.register_value_type::<T, S>(a);
        vec![int_fma::expand::<T, S>(scope, a.into(), b.into(), c.into()).read_value(scope)]
    }
}

#[cube]
fn int_fma<T: Int, N: Size>(a: Vector<T, N>, b: Vector<T, N>, c: Vector<T, N>) -> Vector<T, N> {
    a * b + c
}
