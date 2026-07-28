use super::prelude::*;
use cubecl_core::ir::dialect::general::{CastOp, SelectOp};

fn int_repr(ctx: &Context, ty: TypeHandle) -> Option<(u32, bool)> {
    let ty = ty.deref(ctx);
    if let Some(int) = ty.downcast_ref::<IntegerType>() {
        Some((int.width(), int.signedness() == Signedness::Signed))
    } else if ty.is::<BoolType>() {
        Some((1, false))
    } else if ty.is::<IndexType>() {
        Some((64, false))
    } else {
        None
    }
}

fn cast_int_to_int(
    cast_op: &CastOp,
    is_signed: bool,
    in_width: u32,
    out_width: u32,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) -> Result<()> {
    let out_ty = cast_op.get_result(ctx).get_type(ctx);
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();

    let out_ty = cube_type_to_llvm(ctx, out_ty);

    if out_ty.deref(ctx).is::<BoolType>() && in_width > 1 {
        let zero = insert_int_const(ctx, rewriter, in_width, 0);
        let cmp = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, input, zero);
        rewriter.insert_op(ctx, &cmp);
        rewriter.replace_operation_with_values(ctx, old_op, vec![cmp.get_result(ctx)]);
    } else if in_width == out_width {
        rewriter.replace_operation_with_values(ctx, old_op, vec![input]);
    } else if out_width > in_width {
        if is_signed {
            let op = llvm::SExtOp::new(ctx, input, out_ty);
            rewriter.insert_op(ctx, &op);
            rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
        } else {
            let op = llvm::ZExtOp::new_with_nneg(ctx, input, out_ty, false);
            rewriter.insert_op(ctx, &op);
            rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
        }
    } else {
        let op = llvm::TruncOp::new(ctx, input, out_ty);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    }

    Ok(())
}

fn cast_float_to_int(
    cast_op: &CastOp,
    is_signed: bool,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) {
    let res_ty = cube_type_to_llvm(ctx, cast_op.result_type(ctx));
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();

    if is_signed {
        let op = llvm::FPToSIOp::new(ctx, input, res_ty);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    } else {
        let op = llvm::FPToUIOp::new(ctx, input, res_ty);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    };
}

fn extract_elem_type(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    if let Some(ty) = ty.deref(ctx).downcast_ref::<LlvmVectorType>() {
        ty.elem_type()
    } else if let Some(ty) = ty.deref(ctx).downcast_ref::<CubeVectorType>() {
        ty.scalar_type(ctx)
    } else {
        ty
    }
}

#[op_interface_impl]
impl ToLLVMDialect for CastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let in_ty = self.input(ctx).get_type(ctx);
        let in_ty = extract_elem_type(ctx, in_ty);
        let out_ty = self.get_result(ctx).get_type(ctx);
        let out_ty = extract_elem_type(ctx, out_ty);

        if let (Some((in_width, in_signed)), Some((out_width, _))) =
            (int_repr(ctx, in_ty), int_repr(ctx, out_ty))
        {
            return cast_int_to_int(self, in_signed, in_width, out_width, ctx, rewriter);
        }

        let is_float = |ty: TypeHandle| {
            let ty = ty.deref(ctx);
            ty.is::<FP16Type>() || ty.is::<FP32Type>() || ty.is::<FP64Type>()
        };

        if is_float(in_ty)
            && let Some((_, out_signed)) = int_repr(ctx, out_ty)
        {
            cast_float_to_int(self, out_signed, ctx, rewriter);
        }

        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for SelectOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let new_op = llvm::SelectOp::new(
            ctx,
            self.condition(ctx),
            self.true_value(ctx),
            self.false_value(ctx),
        );
        rewriter.insert_op(ctx, &new_op);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![new_op.get_result(ctx)],
        );
        Ok(())
    }
}
