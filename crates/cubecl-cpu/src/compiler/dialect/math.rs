use cubecl_core::ir::dialect::math::IMulOp;
use pliron_llvm::{
    attributes::IntegerOverflowFlagsAttr, op_interfaces::IntBinArithOpWithOverflowFlag, ops::MulOp,
};

use super::prelude::*;

impl ToLLVMDialect for IMulOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);

        let op = MulOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
        let r = op.get_result(ctx);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![r]);
        Ok(())
    }
}
