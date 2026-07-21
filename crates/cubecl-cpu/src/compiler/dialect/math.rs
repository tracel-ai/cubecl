use cubecl_core::ir::dialect::math::{FAbsOp, IAddOp, IMulOp};
use pliron_llvm::types::FuncType;
use pliron_llvm::{
    attributes::IntegerOverflowFlagsAttr, op_interfaces::IntBinArithOpWithOverflowFlag, ops as llvm,
};

use crate::compiler::dialect::to_llvm::cube_type_to_llvm;

use super::prelude::*;

macro_rules! lower_int_bin_arith {
    ($cube_op:ty => $llvm_op:ty) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let lhs = self.lhs(ctx);
                let rhs = self.rhs(ctx);
                let op = <$llvm_op>::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                );
                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_int_bin_arith!(IMulOp => llvm::MulOp);
lower_int_bin_arith!(IAddOp => llvm::AddOp);

#[op_interface_impl]
impl ToLLVMDialect for FAbsOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let elem_ty = cube_type_to_llvm(ctx, input.get_type(ctx));
        let intrinsic_type = FuncType::get(ctx, elem_ty, vec![elem_ty], false);

        let op = llvm::CallIntrinsicOp::new(ctx, "llvm.fabs".into(), intrinsic_type, vec![input]);

        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}
