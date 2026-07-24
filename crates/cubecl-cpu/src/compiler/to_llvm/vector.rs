use cubecl_core::ir::dialect::vector::{self, VectorBroadcastOp, VectorExtractOp, VectorInsertOp};
use cubecl_core::ir::interfaces::TypedExt;
use cubecl_core::ir::prelude::*;
use pliron::attribute::AttrObj;
use pliron::builtin::attributes::{FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::utils::apfloat::{self, Float};
use pliron::utils::apint::{APInt, bw};
use pliron_llvm::ops::{self as llvm};
use pliron_llvm::types::FuncType;

use crate::compiler::to_llvm::ToLLVMDialect;
use crate::compiler::to_llvm::ty::cube_type_to_llvm;

#[op_interface_impl]
impl ToLLVMDialect for VectorBroadcastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let vec_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));
        let poison = llvm::PoisonOp::new(ctx, vec_ty);
        rewriter.insert_op(ctx, &poison);
        let i32_zero_attr = IntegerAttr::new(
            IntegerType::get(ctx, 32, Signedness::Signless),
            APInt::from_i128(0, bw(32)),
        );
        let zero = llvm::ConstantOp::new(ctx, i32_zero_attr.into());
        rewriter.insert_op(ctx, &zero);
        let num_lanes = self.get_result(ctx).get_type(ctx).vector_size(ctx);

        let scalar = self.input(ctx);
        let inserted =
            llvm::InsertElementOp::new(ctx, poison.get_result(ctx), scalar, zero.get_result(ctx));

        rewriter.insert_op(ctx, &inserted);

        let mask = vec![0; num_lanes];
        let splat =
            llvm::ShuffleVectorOp::new(ctx, inserted.get_result(ctx), poison.get_result(ctx), mask);

        rewriter.insert_op(ctx, &splat);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![splat.get_result(ctx)],
        );

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
        let index = self.index(ctx).0;
        let i32_index_attr = IntegerAttr::new(
            IntegerType::get(ctx, 32, Signedness::Signless),
            APInt::from_i128(index as i128, bw(32)),
        );
        let index_const = llvm::ConstantOp::new(ctx, i32_index_attr.into());
        rewriter.insert_op(ctx, &index_const);

        let inserted = llvm::InsertElementOp::new(
            ctx,
            self.vector(ctx),
            self.value(ctx),
            index_const.get_result(ctx),
        );
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
        let index = self.index(ctx).0;
        let i32_index_attr = IntegerAttr::new(
            IntegerType::get(ctx, 32, Signedness::Signless),
            APInt::from_i128(index as i128, bw(32)),
        );
        let index_const = llvm::ConstantOp::new(ctx, i32_index_attr.into());
        rewriter.insert_op(ctx, &index_const);

        let extracted =
            llvm::ExtractElementOp::new(ctx, self.vector(ctx), index_const.get_result(ctx));
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
        let attr: AttrObj = if res_ty.is_float16(ctx) {
            FPHalfAttr(apfloat::Half::from_bits(0.0f32.to_bits() as u128)).into()
        } else if res_ty.is_float64(ctx) {
            FPDoubleAttr::from(0.0).into()
        } else {
            FPSingleAttr::from(0.0).into()
        };

        let res_ty = cube_type_to_llvm(ctx, res_ty);

        let zero_const = llvm::ConstantOp::new(ctx, attr);
        rewriter.insert_op(ctx, &zero_const);

        let intrinsic_type = FuncType::get(ctx, res_ty, vec![res_ty.into(), elem_ty], false);

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
