use cubecl_ir::{dialect::matrix, interfaces::TypedExt, prelude::*, types::MatrixLayout};
use pliron::{
    builtin::{attributes::IntegerAttr, ops::ConstantOp, types::IntegerType},
    utils::apint::{APInt, bw},
};
use pliron_spirv::{
    ops::{
        CompositeConstructOp, LoadOp, ShiftRightLogicalOp, StoreOp,
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
    let ty = TypedHandle::<CooperativeMatrixType>::from_handle(unwrap_ptr(ty, ctx), ctx).unwrap();
    ty.deref(ctx).use_
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::FillOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let matrix_ptr = self.matrix(ctx);
        let value = self.value(ctx);
        let matrix_ty = ty_to_spirv_dialect(ctx, unwrap_ptr(matrix_ptr, ctx));
        let value = CompositeConstructOp::new(ctx, matrix_ty, vec![value]);
        rewriter.append_op(ctx, &value);
        let value = value.get_result(ctx);
        let store = StoreOp::new(ctx, matrix_ptr, value, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::LoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let matrix_ptr = self.matrix(ctx);
        let value_ptr = self.source(ctx);
        let align = unwrap_ptr(value_ptr, ctx).align(ctx) as u32;
        let layout = layout_to_spirv(self.layout(ctx).0);
        let stride = adjust_stride(ctx, rewriter, self.stride(ctx), value_ptr);
        let matrix_ty = ty_to_spirv_dialect(ctx, unwrap_ptr(matrix_ptr, ctx));

        let value = CooperativeMatrixLoadOp::new(
            ctx,
            matrix_ty,
            value_ptr,
            layout,
            Some(stride),
            MemoryAccess::ALIGNED,
            Some(align),
        );
        rewriter.append_op(ctx, &value);
        let value = value.get_result(ctx);
        let store = StoreOp::new(ctx, matrix_ptr, value, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::StoreOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let matrix_ptr = self.matrix(ctx);
        let value_ptr = self.destination(ctx);
        let align = unwrap_ptr(value_ptr, ctx).align(ctx) as u32;
        let layout = layout_to_spirv(self.layout(ctx).0);
        let stride = adjust_stride(ctx, rewriter, self.stride(ctx), value_ptr);
        let matrix_ty = ty_to_spirv_dialect(ctx, unwrap_ptr(matrix_ptr, ctx));

        let value = LoadOp::new(ctx, matrix_ty, matrix_ptr, MemoryAccess::NONE, None);
        let value = rewriter.append_op_with_result(ctx, &value);
        let store = CooperativeMatrixStoreOp::new(
            ctx,
            value_ptr,
            value,
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
    ctx: &Context,
    value: Value,
    operands_info: &OperandsInfo,
) -> TypeHandle {
    let info = operands_info.lookup_most_recent_type(value).unwrap();
    info.element_ty(ctx)
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::MultiplyAccumulateOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_a = self.mat_a(ctx);
        let mat_b = self.mat_b(ctx);
        let mat_c = self.mat_c(ctx);
        let mat_d = self.mat_d(ctx);

        let mut operands = CooperativeMatrixOperands::NONE_KHR;
        if elem_ty_prev(ctx, mat_a, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_A_SIGNED_COMPONENTS_KHR;
        }
        if elem_ty_prev(ctx, mat_b, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_B_SIGNED_COMPONENTS_KHR;
        }
        if elem_ty_prev(ctx, mat_c, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_C_SIGNED_COMPONENTS_KHR;
        }
        if elem_ty_prev(ctx, mat_d, operands_info).is_signed_int(ctx) {
            operands |= CooperativeMatrixOperands::MATRIX_RESULT_SIGNED_COMPONENTS_KHR;
        }

        let mat_a = LoadOp::new(ctx, unwrap_ptr(mat_a, ctx), mat_a, MemoryAccess::NONE, None);
        let mat_b = LoadOp::new(ctx, unwrap_ptr(mat_b, ctx), mat_b, MemoryAccess::NONE, None);
        let mat_c = LoadOp::new(ctx, unwrap_ptr(mat_c, ctx), mat_c, MemoryAccess::NONE, None);

        let mat_a = rewriter.append_op_with_result(ctx, &mat_a);
        let mat_b = rewriter.append_op_with_result(ctx, &mat_b);
        let mat_c = rewriter.append_op_with_result(ctx, &mat_c);

        let execute = CooperativeMatrixMulAddOp::new(
            ctx,
            unwrap_ptr(mat_d, ctx),
            mat_a,
            mat_b,
            mat_c,
            Some(operands.into()),
        );
        let value_d = rewriter.append_op_with_result(ctx, &execute);
        let store = StoreOp::new(ctx, mat_d, value_d, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::CastOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_in = self.input(ctx);
        let mat_out = self.output(ctx);

        let input_ident = matrix_ident(ctx, mat_in);
        let output_ident = matrix_ident(ctx, mat_out);

        let in_ty = unwrap_ptr(mat_in, ctx);
        let elem_in = elem_ty_prev(ctx, mat_in, operands_info);
        let out_ty = unwrap_ptr(mat_out, ctx);
        let elem_out = elem_ty_prev(ctx, mat_out, operands_info);

        let mat_in = LoadOp::new(ctx, in_ty, mat_in, MemoryAccess::NONE, None);
        let mat_in = rewriter.append_op_with_result(ctx, &mat_in);

        let value_out = if elem_in == elem_out && input_ident != output_ident {
            let cast = CooperativeMatrixConvertOp::new(ctx, out_ty, mat_in);
            rewriter.append_op_with_result(ctx, &cast)
        } else {
            let out_ty_prev = operands_info.lookup_most_recent_type(mat_out).unwrap();
            cast(ctx, rewriter, mat_in, elem_in, out_ty_prev.unwrap_ptr(ctx))
        };

        let store = StoreOp::new(ctx, mat_out, value_out, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for matrix::ElementwiseOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let mat_in = self.matrix_in(ctx);
        let mat_out = self.matrix_out(ctx);
        let closure = self.closure(ctx);
        let captures = self.closure_captures(ctx);

        let in_ty = unwrap_ptr(mat_in, ctx);
        let out_ty = unwrap_ptr(mat_out, ctx);

        let mat_in = LoadOp::new(ctx, in_ty, mat_in, MemoryAccess::NONE, None);
        let mat_in = rewriter.append_op_with_result(ctx, &mat_in);

        let elemwise = CooperativeMatrixPerElementOpOp::new(ctx, out_ty, mat_in, closure, captures);
        let value_out = rewriter.append_op_with_result(ctx, &elemwise);

        let store = StoreOp::new(ctx, mat_out, value_out, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}
