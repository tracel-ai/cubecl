use cubecl_core as cubecl;
use cubecl_core::ir::dialect::math::ArcSinhOp;
use cubecl_core::ir::prelude::*;
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

#[cube]
fn arc_sinh<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    (x + (x * x + Vector::from_int(1)).sqrt()).ln()
}

#[op_interface_impl]
impl LowerOp for ArcSinhOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let value = self.input(scope.ctx());
        scope.register_value_type::<T, S>(value);
        vec![arc_sinh::expand::<T, S>(scope, value.into()).read_value(scope)]
    }
}
