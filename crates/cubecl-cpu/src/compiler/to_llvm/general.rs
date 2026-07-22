use super::ToLLVMDialect;
use cubecl_core::ir::dialect::general::CastOp;
use cubecl_core::ir::interfaces::ScalarizableType;
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::types::VectorType as CubeVectorType;
use cubecl_core::ir::types::scalar::{BoolType, IndexType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness};
use pliron::utils::apint::{APInt, bw};
use pliron_llvm::attributes::ICmpPredicateAttr;
use pliron_llvm::op_interfaces::{CastOpInterface, CastOpWithNNegInterface};
use pliron_llvm::ops::{self as llvm};
use pliron_llvm::types::VectorType as LLVMVectorType;

use crate::compiler::to_llvm::ty::cube_type_to_llvm;

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
        let in_llvm = IntegerType::get(ctx, in_width, Signedness::Signless);
        let zero_attr = IntegerAttr::new(in_llvm, APInt::zero(bw(in_width as usize)));
        let zero = llvm::ConstantOp::new(ctx, zero_attr.into());
        rewriter.insert_op(ctx, &zero);
        let cmp = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, input, zero.get_result(ctx));
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
    if let Some(ty) = ty.deref(ctx).downcast_ref::<LLVMVectorType>() {
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
