use cubecl_ir::{dialect::ssa_matrix, interfaces::TypedExt, prelude::*, types::MatrixLayout};
use pliron::{
    builtin::{attributes::IntegerAttr, ops::ConstantOp, types::IntegerType},
    utils::apint::{APInt, bw},
};
use pliron_spirv::{
    ops::{
        CompositeConstructOp, ShiftRightLogicalOp,
        khr::{CooperativeMatrixLoadOp, CooperativeMatrixMulAddOp, CooperativeMatrixStoreOp},
        nv::{CooperativeMatrixConvertOp, CooperativeMatrixPerElementOpOp},
    },
    types::{PointerType, khr::CooperativeMatrixType},
};
use rspirv::spirv::{
    CooperativeMatrixLayout, CooperativeMatrixOperands, CooperativeMatrixUse, MemoryAccess,
};

use crate::{
    ops::{convert::cast, to_spirv_dialect::ToSpirvDialectOp},
    types::ty_to_spirv_dialect,
};

fn layout_to_spirv(layout: MatrixLayout) -> CooperativeMatrixLayout {
    match layout {
        MatrixLayout::ColMajor => CooperativeMatrixLayout::ColumnMajorKHR,
        MatrixLayout::RowMajor => CooperativeMatrixLayout::RowMajorKHR,
        MatrixLayout::Undefined => CooperativeMatrixLayout::RowMajorKHR,
    }
}

fn adjust_stride(
    ctx: &mut Context,
    rewriter: &mut impl Rewriter,
    stride: Value,
    value_ptr: impl Typed,
) -> Value {
    let vector_size = unwrap_ptr(value_ptr, ctx).vector_size(ctx);
    if vector_size > 1 {
        let ref_ty = TypedHandle::<IntegerType>::from_handle(stride.get_type(ctx), ctx).unwrap();
        let width = bw(ref_ty.deref(ctx).width() as usize);
        let shift = IntegerAttr::new(ref_ty, APInt::from_u32(vector_size.trailing_zeros(), width));
        let shift = ConstantOp::new(ctx, Box::new(shift));
        let shift = rewriter.append_op_with_result(ctx, &shift);
        let shift_op = ShiftRightLogicalOp::new(ctx, stride.get_type(ctx), stride, shift);
        rewriter.append_op_with_result(ctx, &shift_op)
    } else {
        stride
    }
}

pub(super) fn unwrap_ptr(ty: impl Typed, ctx: &Context) -> TypeHandle {
    if let Some(ptr) = ty.get_type(ctx).deref(ctx).downcast_ref::<PointerType>() {
        ptr.element_type
    } else {
        ty.get_type(ctx)
    }
}

fn matrix_ident(ctx: &Context, ty: impl Typed) -> CooperativeMatrixUse {
    let ty = TypedHandle::<CooperativeMatrixType>::from_handle(ty.get_type(ctx), ctx).unwrap();
    ty.deref(ctx).use_
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::FillOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        let matrix_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let value = CompositeConstructOp::new(ctx, matrix_ty, vec![value]);
        rewriter.append_op(ctx, &value);
        rewriter.replace_operation(ctx, self.get_operation(), value.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::LoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value_ptr = self.source(ctx);
        let align = unwrap_ptr(value_ptr, ctx).align(ctx) as u32;
        let layout = layout_to_spirv(self.layout(ctx).0);
        let stride = adjust_stride(ctx, rewriter, self.stride(ctx), value_ptr);
        let matrix_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));

        let load = CooperativeMatrixLoadOp::new(
            ctx,
            matrix_ty,
            value_ptr,
            layout,
            Some(stride),
            MemoryAccess::ALIGNED,
            Some(align),
        );
        rewriter.append_op(ctx, &load);
        rewriter.replace_operation(ctx, self.get_operation(), load.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::StoreOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let matrix = self.matrix(ctx);
        let value_ptr = self.destination(ctx);
        let align = unwrap_ptr(value_ptr, ctx).align(ctx) as u32;
        let layout = layout_to_spirv(self.layout(ctx).0);
        let stride = adjust_stride(ctx, rewriter, self.stride(ctx), value_ptr);

        let store = CooperativeMatrixStoreOp::new(
            ctx,
            value_ptr,
            matrix,
            layout,
            Some(stride),
            MemoryAccess::ALIGNED,
            Some(align),
        );
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

pub(super) fn elem_ty_prev(
    value: Value,
    ctx: &Context,
    operands_info: &OperandsInfo,
) -> TypeHandle {
    let info = operands_info.lookup_most_recent_type(value).unwrap();
    info.element_ty(ctx)
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::MultiplyAccumulateOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_a = self.mat_a(ctx);
        let mat_b = self.mat_b(ctx);
        let mat_c = self.mat_c(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));

        let mut operands = CooperativeMatrixOperands::NONE_KHR;
        if elem_ty_prev(mat_a, ctx, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR;
        }
        if elem_ty_prev(mat_b, ctx, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR;
        }
        if elem_ty_prev(mat_c, ctx, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR;
        }
        if self.result_type(ctx).element_ty(ctx).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR;
        }

        let execute =
            CooperativeMatrixMulAddOp::new(ctx, out_ty, mat_a, mat_b, mat_c, Some(operands.into()));
        rewriter.append_op(ctx, &execute);
        rewriter.replace_operation(ctx, self.get_operation(), execute.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::CastOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_in = self.input(ctx);

        let elem_in = elem_ty_prev(mat_in, ctx, operands_info);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let elem_out = self.result_type(ctx).element_ty(ctx);

        let input_ident = matrix_ident(ctx, mat_in);
        let output_ident = matrix_ident(ctx, out_ty);

        let value_out = if elem_in == elem_out && input_ident != output_ident {
            let cast = CooperativeMatrixConvertOp::new(ctx, out_ty, mat_in);
            rewriter.append_op_with_result(ctx, &cast)
        } else {
            cast(ctx, rewriter, mat_in, elem_in, self.result_type(ctx))
        };

        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![value_out]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for ssa_matrix::ElementwiseOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_in = self.matrix_in(ctx);
        let closure = self.closure(ctx);
        let captures = self.closure_captures(ctx);

        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));

        let elemwise = CooperativeMatrixPerElementOpOp::new(ctx, out_ty, mat_in, closure, captures);
        rewriter.append_op(ctx, &elemwise);
        rewriter.replace_operation(ctx, self.get_operation(), elemwise.get_operation());
        Ok(())
    }
}
