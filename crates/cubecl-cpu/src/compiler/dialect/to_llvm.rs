use cubecl_core::ir::{Scope, prelude::*};

#[op_interface]
pub trait LowerOpLLVM {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerOpsLLVMPass = MatchRewritePass<LowerOpsLLVM>;

#[derive(Default, Clone, Copy)]
pub struct LowerOpsLLVM;

impl MatchRewrite for LowerOpsLLVM {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOpLLVM>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOpLLVM>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}
