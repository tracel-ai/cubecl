use pliron::builtin::{
    attributes::IntegerAttr,
    ops::ConstantOp,
    types::{IntegerType, Signedness},
};
use pliron_llvm::ops as llvm;

use super::prelude::*;

#[op_interface_impl]
impl ToLLVMDialect for ConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_value(ctx);
        let Some(int_attr) = value.downcast_ref::<IntegerAttr>() else {
            return Ok(());
        };
        let ty = int_attr.get_type();
        let val = int_attr.value();
        let width = ty.deref(ctx).width();
        let const_value = IntegerAttr::new(IntegerType::get(ctx, width, Signedness::Signless), val);

        let llvm_const = llvm::ConstantOp::new(ctx, const_value.into());

        rewriter.insert_operation(ctx, llvm_const.get_operation());

        let old_op = self.get_operation();
        rewriter.replace_operation(ctx, old_op, llvm_const.get_operation());

        Ok(())
    }
}
