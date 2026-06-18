use alloc::vec;
use cubecl_ir::{
    dialect::{
        base::OperationPtrExt,
        general::{CastOp, CopyOp, ReinterpretCastOp},
    },
    prelude::*,
    verify_op_succ,
};

/// Operation that's equivalent to a trivial value copy if input and output types match
#[op_interface]
pub trait TrivialOp: NOpdsInterface<1> + OneResultInterface {
    verify_op_succ!();
}

#[op_interface_impl]
impl TrivialOp for CopyOp {}
#[op_interface_impl]
impl TrivialOp for CastOp {}
#[op_interface_impl]
impl TrivialOp for ReinterpretCastOp {}

pub struct RemoveTrivialOpsPass;

impl DialectConversion for RemoveTrivialOpsPass {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn TrivialOp>(ctx)
            && op.operand(ctx, 0).get_type(ctx) == op.result(ctx).get_type(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.replace_operation_with_values(ctx, op, vec![op.operand(ctx, 0)]);
        Ok(())
    }
}
