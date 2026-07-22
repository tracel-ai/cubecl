use super::ToLLVMDialect;
use cubecl_core::ir::attributes::IndexAttr;
use cubecl_core::ir::prelude::*;
use pliron::builtin::{
    attributes::IntegerAttr,
    ops::ConstantOp,
    types::{IntegerType, Signedness},
};
use pliron::utils::apint::{APInt, bw};
use pliron_llvm::ops as llvm;

use crate::compiler::to_llvm::ty::INDEX_WIDTH;

#[op_interface_impl]
impl ToLLVMDialect for ConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_value(ctx);

        let const_value = if let Some(int_attr) = value.downcast_ref::<IntegerAttr>() {
            let width = int_attr.get_type().deref(ctx).width();
            IntegerAttr::new(
                IntegerType::get(ctx, width, Signedness::Signless),
                int_attr.value(),
            )
        } else if let Some(index_attr) = value.downcast_ref::<IndexAttr>() {
            IntegerAttr::new(
                IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless),
                APInt::from_u64(index_attr.0 as u64, bw(INDEX_WIDTH as usize)),
            )
        } else {
            return Ok(());
        };

        let llvm_const = llvm::ConstantOp::new(ctx, const_value.into());
        rewriter.insert_operation(ctx, llvm_const.get_operation());

        let old_op = self.get_operation();
        rewriter.replace_operation(ctx, old_op, llvm_const.get_operation());

        Ok(())
    }
}
