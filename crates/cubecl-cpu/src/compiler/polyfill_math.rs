use cubecl_core as cubecl;
use cubecl_core::ir::dialect::math::{
    ArcCoshOp, ArcSinhOp, ArcTanhOp, DegreesOp, ErfOp, Expm1Op, Log1pOp, RadiansOp, RecipOp,
    RsqrtOp, SNegOp,
};
use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::polyfills::{erf, expm1, log1p, recip, to_degrees, to_radians};
use cubecl_core::prelude::*;

use cubecl_core::ir::{Scope, dialect::base::OperationPtrExt};

#[op_interface]
pub trait LowerOp {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerComplexMathPass = MatchRewritePass<LowerComplexMath>;

#[derive(new, Default, Clone, Copy)]
pub struct LowerComplexMath;

impl MatchRewrite for LowerComplexMath {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOp>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOp>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        transfer_result_names(ctx, op, &new_values);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}

macro_rules! lower_unary_math_arith {
    ($cube_op:ty => $polyfill:ident) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let value = self.input(scope.ctx());
                scope.register_value_type::<T, S>(value);
                vec![$polyfill::expand::<T, S>(scope, value.into()).read_value(scope)]
            }
        }
    };
}

#[cube]
fn arc_sinh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    (x + (x * x + Vector::one()).sqrt()).ln()
}

lower_unary_math_arith!(ArcSinhOp => arc_sinh);

#[cube]
fn arc_cosh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    (x + (x * x - Vector::one()).sqrt()).ln()
}

lower_unary_math_arith!(ArcCoshOp => arc_cosh);

#[cube]
fn arc_tanh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    Vector::new(F::new(0.5f32)) * ((Vector::one() + x) / (Vector::one() - x)).ln()
}

lower_unary_math_arith!(ArcTanhOp => arc_tanh);

lower_unary_math_arith!(DegreesOp => to_degrees);
lower_unary_math_arith!(RadiansOp => to_radians);
lower_unary_math_arith!(Log1pOp => log1p);
lower_unary_math_arith!(Expm1Op => expm1);

#[cube]
fn inverse_sqrt<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    Vector::one() / x.sqrt()
}

lower_unary_math_arith!(RsqrtOp => inverse_sqrt);

lower_unary_math_arith!(ErfOp => erf);
lower_unary_math_arith!(RecipOp => recip);

#[cube]
fn neg<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    Vector::zero() - x
}

lower_unary_math_arith!(SNegOp => neg);
