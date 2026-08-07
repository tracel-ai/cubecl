use cubecl_core::ir::{Scope, dialect::general::ReadBuiltinOp, prelude::*};

use crate::{
    shared::builtin::{LowerBuiltins, SharedBuiltin},
    target::Hip,
};

impl MatchRewrite for LowerBuiltins<Hip> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op.as_op::<ReadBuiltinOp>(ctx).unwrap().builtin(ctx).0;
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        if let Some(new_value) = builtin.maybe_lower_shared(&scope) {
            rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        }
        Ok(())
    }
}
