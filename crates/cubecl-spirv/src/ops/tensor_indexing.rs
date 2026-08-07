use cubecl_ir::{
    ConstantValue,
    dialect::spirv,
    interfaces::{ScalarType, TypedExt},
    prelude::*,
    try_cast_ty,
    types::spirv::{ClampMode, TensorLayoutType},
};
use pliron::{
    builtin::{attributes::VecAttr, ops::ConstantOp},
    op::Op,
};
use pliron_spirv::{
    attrs::CompositeAttr,
    ops::{
        StoreOp,
        nv::{
            CreateTensorLayoutOp, CreateTensorViewOp, TensorLayoutSetClampValueOp,
            TensorLayoutSetDimensionOp, TensorLayoutSetStrideOp, TensorLayoutSliceOp,
        },
    },
    tensor_addressing_nv::{CooperativeMatrixLoadTensorOp, CooperativeMatrixStoreTensorOp},
};
use rspirv::spirv::{MemoryAccess, TensorAddressingOperands};

use crate::{
    attributes::attr_to_spirv_dialect,
    ops::{
        builtin::const_op_int32,
        matrix::{elem_ty_prev, unwrap_ptr},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

#[op_interface_impl]
impl ToSpirvDialectOp for spirv::CreateLayoutOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ty = TypedHandle::<TensorLayoutType>::from_handle(self.result_type(ctx), ctx).unwrap();
        let clamp_mode = ty.deref(ctx).clamp_mode;
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let layout = CreateTensorLayoutOp::new(ctx, out_ty);
        let layout = rewriter.append_op_with_result(ctx, &layout);
        let set_dims = TensorLayoutSetDimensionOp::new(ctx, out_ty, layout, self.shape(ctx));
        let mut layout = rewriter.append_op_with_result(ctx, &set_dims);
        if let Some(strides) = self.strides(ctx) {
            let set_strides = TensorLayoutSetStrideOp::new(ctx, out_ty, layout, strides);
            layout = rewriter.append_op_with_result(ctx, &set_strides);
        }
        if let ClampMode::Constant(val) = clamp_mode
            && val != 0
        {
            let value = const_op_int32(ctx, val);
            let value = rewriter.append_op_with_result(ctx, &value);
            let set_clamp_val = TensorLayoutSetClampValueOp::new(ctx, out_ty, layout, value);
            layout = rewriter.append_op_with_result(ctx, &set_clamp_val);
        }
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![layout]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for spirv::CreateViewOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let view = CreateTensorViewOp::new(ctx, out_ty);
        rewriter.append_op(ctx, &view);
        rewriter.replace_operation(ctx, self.get_operation(), view.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for spirv::SliceOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let layout = self.layout(ctx);
        let args = self.offsets(ctx).into_iter().zip(self.shape(ctx));
        let args = args.flat_map(|(offs, shape)| vec![offs, shape]).collect();
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));

        let slice = TensorLayoutSliceOp::new(ctx, out_ty, layout, args);
        rewriter.append_op(ctx, &slice);
        rewriter.replace_operation(ctx, self.get_operation(), slice.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for spirv::LoadTensorOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let buffer = self.buffer(ctx);
        let layout = self.layout(ctx);
        let view = self.view(ctx);
        let mat_out = self.out_mat(ctx);

        let matrix_ty = ty_to_spirv_dialect(ctx, unwrap_ptr(mat_out, ctx));
        let elem_ty = elem_ty_prev(ctx, mat_out, operands_info);
        let align = elem_ty.align(ctx) as u32;
        let zero = const_matrix(ctx, rewriter, elem_ty, matrix_ty, 0.into());

        let operands = match view {
            Some(_) => TensorAddressingOperands::TENSOR_VIEW,
            None => TensorAddressingOperands::NONE,
        };

        let value = CooperativeMatrixLoadTensorOp::new(
            ctx,
            matrix_ty,
            buffer,
            zero,
            layout,
            MemoryAccess::ALIGNED,
            Some(align),
            operands,
            view,
            None,
        );
        let value = rewriter.append_op_with_result(ctx, &value);
        let store = StoreOp::new(ctx, mat_out, value, MemoryAccess::NONE, None);
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for spirv::StoreTensorOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let matrix = self.matrix(ctx);
        let buffer = self.buffer(ctx);
        let layout = self.layout(ctx);
        let view = self.view(ctx);

        let elem_ty = elem_ty_prev(ctx, matrix, operands_info);
        let align = elem_ty.align(ctx) as u32;

        let operands = match view {
            Some(_) => TensorAddressingOperands::TENSOR_VIEW,
            None => TensorAddressingOperands::NONE,
        };

        let store = CooperativeMatrixStoreTensorOp::new(
            ctx,
            buffer,
            matrix,
            layout,
            MemoryAccess::ALIGNED,
            Some(align),
            operands,
            view,
        );
        rewriter.append_op(ctx, &store);
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

fn const_matrix(
    ctx: &mut Context,
    rewriter: &mut impl Rewriter,
    scalar_ty: TypeHandle,
    composite_ty: TypeHandle,
    value: ConstantValue,
) -> Value {
    let elem_ty = {
        let scalar_ty = scalar_ty.deref(ctx);
        try_cast_ty!(scalar_ty, ctx, dyn ScalarType).elem_type(ctx)
    };
    let val = value.cast_to(elem_ty);
    let val = attr_to_spirv_dialect(ctx, &val.as_attribute(ctx, elem_ty));
    let attr = CompositeAttr::new(VecAttr(vec![val]), composite_ty);
    let constant = ConstantOp::new(ctx, Box::new(attr));
    rewriter.append_op_with_result(ctx, &constant)
}
