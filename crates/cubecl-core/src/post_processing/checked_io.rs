use alloc::vec;

use alloc::string::String;
use cubecl_ir::{
    Scope, dialect::memory::IndexOp, prelude::*, settings::ExecutionMode, types::RuntimeArrayType,
};

use crate::io::*;

pub type CheckedIoPass = MatchRewritePass<CheckedIo>;

#[derive(new)]
pub struct CheckedIo {
    mode: ExecutionMode,
    kernel_name: String,
}

impl MatchRewrite for CheckedIo {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        Operation::get_op::<IndexOp>(op, ctx)
            .is_some_and(|it| it.checked(ctx) && is_runtime_array(ctx, it.base(ctx)))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let index = Operation::get_op::<IndexOp>(op, ctx).unwrap();

        let scope = Scope::from_context_and_inserter(ctx, rewriter);

        let new_value = match self.mode {
            ExecutionMode::Checked => {
                expand_checked_index(&scope, index.base(ctx), index.index(ctx))
            }
            ExecutionMode::Validate => {
                expand_validate_index(&scope, index.base(ctx), index.index(ctx), &self.kernel_name)
            }
            ExecutionMode::Unchecked => index.get_result(ctx),
        };
        rewriter.replace_operation_with_values(ctx, index.get_operation(), vec![new_value]);
        Ok(())
    }
}

fn is_runtime_array(ctx: &Context, value: Value) -> bool {
    let ty = value.get_type(ctx).deref(ctx);
    ty.downcast_ref::<RuntimeArrayType>().is_some()
}
