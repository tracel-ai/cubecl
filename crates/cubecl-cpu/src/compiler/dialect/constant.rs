use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::ops::ConstantOp;
use pliron::builtin::types::{IntegerType, Signedness};

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
        let (width, needs_conversion) = {
            let ty = ty.deref(ctx);
            (ty.width(), ty.signedness() != Signedness::Signless)
        };
        if !needs_conversion {
            return Ok(());
        }

        let signless = IntegerType::get(ctx, width, Signedness::Signless);
        let new_const = ConstantOp::new(ctx, IntegerAttr::new(signless, val).into());
        rewriter.insert_op(ctx, &new_const);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![new_const.get_result(ctx)],
        );
        Ok(())
    }
}
