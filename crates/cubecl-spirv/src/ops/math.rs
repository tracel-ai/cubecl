use cubecl_core::{
    self as cubecl,
    num_traits::{One, Zero},
    prelude::*,
};
use cubecl_ir::{dialect::math, prelude::*};
use pliron_spirv::{ext::gl, ops, types::StructType};

use crate::{
    lower::{LowerOp, lower_binop, lower_unop},
    ops::{
        base::{binop_to_spirv_dialect, unop_to_spirv_dialect},
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
binop_to_spirv_dialect!(math::SRemOp => ops::SRemOp);
binop_to_spirv_dialect!(math::URemOp => ops::UModOp);
binop_to_spirv_dialect!(math::FRemOp => ops::FRemOp);
binop_to_spirv_dialect!(math::SModFloorOp => ops::SModOp);
binop_to_spirv_dialect!(math::FModFloorOp => ops::FModOp);

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
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::SMulExtendedOp::new(ctx, out_st, lhs, rhs);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
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
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::UMulExtendedOp::new(ctx, out_st, lhs, rhs);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
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
lower_unop!(math::ErfOp, erf);
lower_binop!(math::HypotOp, hypot);
lower_binop!(math::RhypotOp, rhypot);

lower_binop!(math::PowfOp, powf);
lower_binop!(math::PowiOp, powi);

#[cube]
fn log1p<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    (input + Vector::one()).ln()
}

#[cube]
fn expm1<T: Float, N: Size>(x: Vector<T, N>) -> Vector<T, N> {
    let sq = x * x;
    let a = sq * Vector::new(T::new(0.5));
    let b = sq * x * Vector::new(T::new(1.0 / 6.0));
    let taylor = x + a + b;
    let is_small = x.abs().less_than(&Vector::new(T::new(1e-5)));
    select_many(is_small, taylor, x.exp() - Vector::one())
}

#[cube]
fn recip<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::one() / input
}

/// use the SPIR-V dialect version because otherwise we'd get an infinite lowering loop
#[cube]
fn spirv_pow<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let base = base.read_value(scope);
        let exp = exp.read_value(scope);
        let ty = base.get_type(scope.ctx());
        let powf = gl::PowOp::new(scope.ctx_mut(), ty, base, exp);
        scope.register_with_result(&powf).into()
    })
}

#[cube]
fn powf<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    let modulo = exp.mod_floor(Vector::new(T::new(2.0)));
    let is_even = modulo.equal(&Vector::zero());
    let is_odd = modulo.equal(&Vector::one());
    let is_neg_base = base.less_than(&Vector::zero());

    let even_res = spirv_pow(base.abs(), exp);
    let odd_neg_res = -(spirv_pow(-base, exp));
    let default = spirv_pow(base, exp);

    let sel1 = select_many(is_odd.vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

#[cube]
fn powi<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let is_even = exp.is_multiple_of(2);
    let is_neg_base = base.less_than(&Vector::zero());
    let exp = Vector::cast_from(exp);

    let even_res = spirv_pow(base.abs(), exp);
    let odd_neg_res = -(spirv_pow(-base, exp));
    let default = spirv_pow(base, exp);

    let sel1 = select_many((!is_even).vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

#[op_interface_impl]
impl LowerOp for math::FmaOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let a = self.a(scope.ctx());
        let b = self.b(scope.ctx());
        let c = self.c(scope.ctx());
        scope.register_value_type::<T, S>(a);
        vec![fma::expand::<T, S>(scope, a.into(), b.into(), c.into()).read_value(scope)]
    }
}

#[cube]
fn fma<T: Float, N: Size>(a: Vector<T, N>, b: Vector<T, N>, c: Vector<T, N>) -> Vector<T, N> {
    a * b + c
}
