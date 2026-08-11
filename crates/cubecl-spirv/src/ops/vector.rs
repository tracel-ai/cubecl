use cubecl_core::{self as cubecl, define_scalar, define_size, num_traits::One, prelude::*};
use cubecl_ir::{dialect::vector, interfaces::TypedExt, prelude::*};
use pliron_spirv::{
    ext::gl,
    ops::{self, CompositeConstructOp, CompositeExtractOp, CompositeInsertOp},
};

use crate::{
    lower::lower_unop,
    ops::{
        base::{binop_to_spirv_dialect, unop_to_spirv_dialect},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorInitOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let opds = op.operands(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = CompositeConstructOp::new(ctx, out_ty, opds);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorBroadcastOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let opds = vec![self.input(ctx); self.result_type(ctx).vector_size(ctx)];
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = CompositeConstructOp::new(ctx, out_ty, opds);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorInsertOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let vector = self.vector(ctx);
        let idx = self.index(ctx).0 as u32;
        let value = self.value(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = CompositeInsertOp::new(ctx, out_ty, value, vector, vec![idx.into()]);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorExtractOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let vector = self.vector(ctx);
        let idx = self.index(ctx).0 as u32;
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = CompositeExtractOp::new(ctx, out_ty, vector, vec![idx.into()]);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorInsertDynamicOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let vector = self.vector(ctx);
        let idx = self.index(ctx);
        let value = self.value(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = ops::VectorInsertDynamicOp::new(ctx, out_ty, vector, value, idx);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for vector::VectorExtractDynamicOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let vector = self.vector(ctx);
        let idx = self.index(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let new_op = ops::VectorExtractDynamicOp::new(ctx, out_ty, vector, idx);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

unop_to_spirv_dialect!(vector::MagnitudeOp => gl::LengthOp);
unop_to_spirv_dialect!(vector::NormalizeOp => gl::NormalizeOp);
binop_to_spirv_dialect!(vector::SDotOp => ops::SDotOp, None);
binop_to_spirv_dialect!(vector::UDotOp => ops::UDotOp, None);
binop_to_spirv_dialect!(vector::FDotOp => ops::DotOp);

lower_unop!(vector::ISumOp, i_vector_sum);
lower_unop!(vector::FSumOp, f_vector_sum);

#[cube]
fn i_vector_sum<T: Numeric, N: Size>(vector: Vector<T, N>) -> T {
    let mut out = vector.extract(0usize);
    #[unroll]
    for i in 1..vector.vector_size() {
        out += vector.extract(i);
    }
    out
}

#[cube]
fn f_vector_sum<T: Float, N: Size>(vector: Vector<T, N>) -> T {
    vector.dot(Vector::one())
}
