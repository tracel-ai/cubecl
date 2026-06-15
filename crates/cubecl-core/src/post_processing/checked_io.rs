use alloc::vec;

use alloc::string::String;
use cubecl_ir::{
    Scope,
    dialect::memory::IndexOp,
    pliron::{
        builtin::op_interfaces::OneResultInterface,
        context::{Context, Ptr},
        irbuild::{
            dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
            rewriter::Rewriter,
        },
        op::Op,
        operation::Operation,
        prelude::Result,
        r#type::Typed,
        value::Value,
    },
    types::RuntimeArrayType,
};
use cubecl_runtime::server::ExecutionMode;

use crate::io::*;

pub struct ApplyCheckedIo {
    mode: ExecutionMode,
    kernel_name: String,
}

impl DialectConversion for ApplyCheckedIo {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        Operation::get_op::<IndexOp>(op, ctx).is_some_and(|it| {
            it.get_attr_checked(ctx).unwrap().0 && is_runtime_array(ctx, it.base(ctx))
        })
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let index = Operation::get_op::<IndexOp>(op, ctx).unwrap();

        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let unroll_factor = index.get_attr_unroll_factor(ctx).unwrap().0;

        let new_value = match self.mode {
            ExecutionMode::Checked => {
                expand_checked_index(&scope, index.base(ctx), index.index(ctx), unroll_factor)
            }
            ExecutionMode::Validate => expand_validate_index(
                &scope,
                index.base(ctx),
                index.index(ctx),
                unroll_factor,
                &self.kernel_name,
            ),
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
