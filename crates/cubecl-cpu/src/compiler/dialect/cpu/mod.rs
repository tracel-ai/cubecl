use cubecl_core::ir::{Scope, prelude::*};

pub mod entrypoint;

pub use entrypoint::InsertConstantEmulationPass;

#[op_interface]
pub trait LowerOpCpu {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerOpsCpuPass = MatchRewritePass<LowerOpsCpu>;

#[derive(Default, Clone, Copy)]
pub struct LowerOpsCpu;

impl MatchRewrite for LowerOpsCpu {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOpCpu>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOpCpu>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}

#[op_interface]
pub trait LowerOpConstant {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerOpsConstantPass = MatchRewritePass<LowerOpsConstant>;

#[derive(Default, Clone, Copy)]
pub struct LowerOpsConstant;

impl MatchRewrite for LowerOpsConstant {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOpConstant>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOpConstant>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}
