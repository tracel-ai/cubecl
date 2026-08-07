use super::prelude::*;
use cubecl_core::ir::dialect::cmp::*;

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
                let op = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::$pred, lhs, rhs);
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

/// Lower an float comparison op to `llvm.fcmp` with the given predicate.
macro_rules! lower_float_cmp {
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
                let op = llvm::FCmpOp::new(ctx, FCmpPredicateAttr::$pred, lhs, rhs);
                op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
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

lower_float_cmp!(FLessThanOp => OLT);
lower_float_cmp!(FLessThanOrEqualOp => OLE);
lower_float_cmp!(FGreaterThanOp => OGT);
lower_float_cmp!(FGreaterThanOrEqualOp => OGE);
lower_float_cmp!(FEqualOp => OEQ);
lower_float_cmp!(FNotEqualOp => UNE);
