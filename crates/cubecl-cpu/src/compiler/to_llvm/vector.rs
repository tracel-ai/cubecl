use super::prelude::*;
use cubecl_core::ir::dialect::vector::{
    self, VectorBroadcastOp, VectorExtractOp, VectorInsertDynamicOp, VectorInsertOp,
};

/// Broadcast `scalar` to every lane of `vec_ty`, with the poison/insertelement/shufflevector idiom
/// LLVM folds back into a splat.
pub fn insert_splat(
    ctx: &mut Context,
    rewriter: &mut impl Inserter,
    vec_ty: TypeHandle,
    scalar: Value,
    num_lanes: usize,
) -> Value {
    let poison = llvm::PoisonOp::new(ctx, vec_ty);
    rewriter.insert_op(ctx, &poison);
    let zero = insert_i32_const(ctx, rewriter, 0);

    let inserted = llvm::InsertElementOp::new(ctx, poison.get_result(ctx), scalar, zero);
    rewriter.insert_op(ctx, &inserted);

    let mask = vec![0; num_lanes];
    let splat =
        llvm::ShuffleVectorOp::new(ctx, inserted.get_result(ctx), poison.get_result(ctx), mask);
    rewriter.insert_op(ctx, &splat);

    splat.get_result(ctx)
}

#[op_interface_impl]
impl ToLLVMDialect for VectorBroadcastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let res_ty = self.get_result(ctx).get_type(ctx);
        let num_lanes = res_ty.vector_size(ctx);
        let vec_ty = cube_type_to_llvm(ctx, res_ty);

        let scalar = self.input(ctx);
        let splat = insert_splat(ctx, rewriter, vec_ty, scalar, num_lanes);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![splat]);

        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for VectorInsertOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let index = self.index(ctx).0 as i32;
        let index = insert_i32_const(ctx, rewriter, index);

        let inserted = llvm::InsertElementOp::new(ctx, self.vector(ctx), self.value(ctx), index);
        rewriter.insert_op(ctx, &inserted);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![inserted.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for VectorInsertDynamicOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let inserted =
            llvm::InsertElementOp::new(ctx, self.vector(ctx), self.value(ctx), self.index(ctx));
        rewriter.insert_op(ctx, &inserted);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![inserted.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for VectorExtractOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let index = self.index(ctx).0 as i32;
        let index = insert_i32_const(ctx, rewriter, index);

        let extracted = llvm::ExtractElementOp::new(ctx, self.vector(ctx), index);
        rewriter.insert_op(ctx, &extracted);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![extracted.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for vector::FSumOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let elem_ty = input.get_type(ctx);
        let res_ty = self.get_result(ctx).get_type(ctx);
        // `llvm.vector.reduce.fadd` takes the accumulator start value as its first argument.
        let attr = float_attr(ctx, res_ty, 0.0).unwrap_or_else(|| FPSingleAttr::from(0.0).into());

        let res_ty = cube_type_to_llvm(ctx, res_ty);

        let zero_const = llvm::ConstantOp::new(ctx, attr);
        rewriter.insert_op(ctx, &zero_const);

        let intrinsic_type = FuncType::get(ctx, res_ty, vec![res_ty, elem_ty], false);

        let op = llvm::CallIntrinsicOp::new(
            ctx,
            "llvm.vector.reduce.fadd".into(),
            intrinsic_type,
            vec![zero_const.get_result(ctx), input],
        );

        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for vector::ISumOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let elem_ty = input.get_type(ctx);
        let res_ty = self.get_result(ctx).get_type(ctx);

        let res_ty = cube_type_to_llvm(ctx, res_ty);

        let intrinsic_type = FuncType::get(ctx, res_ty, vec![elem_ty], false);

        let op = llvm::CallIntrinsicOp::new(
            ctx,
            "llvm.vector.reduce.add".into(),
            intrinsic_type,
            vec![input],
        );

        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}
