use super::ToLLVMDialect;
use cubecl_core::ir::attributes::{FloatAttr, IndexAttr};
use cubecl_core::ir::interfaces::TypedExt;
use cubecl_core::ir::prelude::*;
use half::f16;
use pliron::attribute::AttrObj;
use pliron::builtin::attributes::{FPDoubleAttr, FPHalfAttr, FPSingleAttr};
use pliron::builtin::{
    attributes::IntegerAttr,
    ops::ConstantOp,
    types::{IntegerType, Signedness},
};
use pliron::utils::apfloat::{self, Float};
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

        println!("{:?}", value);
        let const_value: AttrObj = if let Some(int_attr) = value.downcast_ref::<IntegerAttr>() {
            let width = int_attr.get_type().deref(ctx).width();
            IntegerAttr::new(
                IntegerType::get(ctx, width, Signedness::Signless),
                int_attr.value(),
            )
            .into()
        } else if let Some(index_attr) = value.downcast_ref::<IndexAttr>() {
            IntegerAttr::new(
                IntegerType::get(ctx, INDEX_WIDTH, Signedness::Signless),
                APInt::from_u64(index_attr.0 as u64, bw(INDEX_WIDTH as usize)),
            )
            .into()
        } else if let Some(float_attr) = value.downcast_ref::<FloatAttr>() {
            if self.get_result(ctx).is_float16(ctx) {
                let val = f16::from_f64(apfloat::double_to_f64(float_attr.val));
                FPHalfAttr(apfloat::Half::from_bits(val.to_bits() as u128)).into()
            } else if self.get_result(ctx).is_float32(ctx) {
                let val = apfloat::double_to_f64(float_attr.val) as f32;
                FPSingleAttr(apfloat::f32_to_single(val)).into()
            } else if self.get_result(ctx).is_float64(ctx) {
                FPDoubleAttr(float_attr.val).into()
            } else {
                return Ok(());
            }
        } else {
            return Ok(());
        };

        let llvm_const = llvm::ConstantOp::new(ctx, const_value);
        rewriter.insert_operation(ctx, llvm_const.get_operation());

        let old_op = self.get_operation();
        rewriter.replace_operation(ctx, old_op, llvm_const.get_operation());

        Ok(())
    }
}
