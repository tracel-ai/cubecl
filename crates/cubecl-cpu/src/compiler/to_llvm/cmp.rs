use super::ToLLVMDialect;
use cubecl_core::ir::dialect::cmp::{BoolEqualOp, BoolNotEqualOp};
use cubecl_core::ir::dialect::cmp::{
    IEqualOp, INotEqualOp, SGreaterThanOp, SGreaterThanOrEqualOp, SLessThanOp, SLessThanOrEqualOp,
    UGreaterThanOp, UGreaterThanOrEqualOp, ULessThanOp, ULessThanOrEqualOp,
};
use cubecl_core::ir::prelude::*;
use pliron_llvm::attributes::ICmpPredicateAttr;
use pliron_llvm::ops::ICmpOp;

/// Lower an integer/index comparison op to `llvm.icmp` with the given predicate.
macro_rules! lower_int_cmp {
    ($cube_op:ty => $pred:ident) => {
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
                let op = ICmpOp::new(ctx, ICmpPredicateAttr::$pred, lhs, rhs);
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

lower_int_cmp!(SLessThanOp => SLT);
lower_int_cmp!(ULessThanOp => ULT);
lower_int_cmp!(SGreaterThanOp => SGT);
lower_int_cmp!(UGreaterThanOp => UGT);
lower_int_cmp!(SLessThanOrEqualOp => SLE);
lower_int_cmp!(ULessThanOrEqualOp => ULE);
lower_int_cmp!(SGreaterThanOrEqualOp => SGE);
lower_int_cmp!(UGreaterThanOrEqualOp => UGE);
lower_int_cmp!(IEqualOp => EQ);
lower_int_cmp!(INotEqualOp => NE);
lower_int_cmp!(BoolEqualOp => EQ);
lower_int_cmp!(BoolNotEqualOp => NE);
