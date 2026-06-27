use core::marker::PhantomData;

use cubecl_core::ir::{Scope, dialect::base::OperationPtrExt, prelude::*};

use crate::target::Shared;

#[op_interface]
pub trait LowerOp<T = Shared> {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerOpsCppPass<T> = MatchRewritePass<LowerOpsCpp<T>>;

#[derive(new, Default, Clone, Copy)]
pub struct LowerOpsCpp<T> {
    _target: PhantomData<T>,
}

impl<T: 'static> MatchRewrite for LowerOpsCpp<T> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOp<T>>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOp<T>>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}
