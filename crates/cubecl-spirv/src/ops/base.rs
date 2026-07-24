macro_rules! unop_to_spirv_dialect {
    ($ty: ty => $new_ty: ty $(,$extra:expr)*) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let op = self.get_operation();
                let inp = op.operand(ctx, 0);
                let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, inp, $($extra),*);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}
pub(crate) use unop_to_spirv_dialect;

macro_rules! binop_to_spirv_dialect {
    ($ty: ty => $new_ty: ty $(,$extra:expr)*) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let op = self.get_operation();
                let lhs = op.operand(ctx, 0);
                let rhs = op.operand(ctx, 1);
                let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, lhs, rhs, $($extra),*);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}
pub(crate) use binop_to_spirv_dialect;
