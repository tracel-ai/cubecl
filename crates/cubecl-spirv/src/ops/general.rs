use cubecl_ir::{
    Scope,
    dialect::{
        general::{self, BufferLenOp, ReadScalarOp, ShapeOp, StrideOp},
        math::IAddOp,
    },
    interfaces::{ScalarType, TypedExt},
    metadata::Info,
    prelude::*,
    try_cast_ty,
};
use pliron_spirv::{
    ext::printf,
    ops::{self, InBoundsAccessChainOp, LoadOp},
    types::PointerType,
};
use rspirv::spirv::{MemoryAccess, StorageClass};

use crate::{
    LowerInfoOp,
    ops::{
        base::{binop_to_spirv_dialect, unop_to_spirv_dialect},
        builtin::const_int32,
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

unop_to_spirv_dialect!(general::CopyOp => ops::CopyObjectOp);
binop_to_spirv_dialect!(general::BoolAndOp => ops::LogicalAndOp);
binop_to_spirv_dialect!(general::BoolOrOp => ops::LogicalOrOp);
unop_to_spirv_dialect!(general::BoolNotOp => ops::LogicalNotOp);
unop_to_spirv_dialect!(general::ReinterpretCastOp => ops::BitcastOp);

#[op_interface_impl]
impl ToSpirvDialectOp for general::SelectOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let cond = self.condition(ctx);
        let true_value = self.true_value(ctx);
        let false_value = self.false_value(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op = ops::SelectOp::new(ctx, out_ty, cond, true_value, false_value);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

macro_rules! erase_op {
    ($ty: ty) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                rewriter.erase_operation(ctx, self.get_operation());
                Ok(())
            }
        }
    };
}

erase_op!(general::FreeOp);
erase_op!(general::CommentOp);

#[op_interface_impl]
impl ToSpirvDialectOp for general::PrintfOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let format_str = self.format_string(ctx).as_str().to_owned();
        let args = self.args(ctx);
        let new_op = printf::DebugPrintfOp::new(ctx, format_str, args);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl LowerInfoOp for ReadScalarOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter, info_st: Value) -> Value {
        let scalars = ctx.aux_ty::<Info>().scalars.clone();
        let id = self.id(ctx).0;
        let ty = self.ty(ctx).get_type(ctx);
        let elem = try_cast_ty!(ty.deref(ctx), ctx, dyn ScalarType).elem_type(ctx);
        let field = scalars.iter().position(|s| s.ty == elem).unwrap();

        load_static_info(ctx, rewriter, info_st, field, ty, id)
    }
}

#[op_interface_impl]
impl LowerInfoOp for BufferLenOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter, info_st: Value) -> Value {
        let field = static_meta_field(ctx);
        let ty = self.result_type(ctx);
        let buffer_id = self.buffer_idx(ctx).0;
        let buf_offs = ctx.aux_ty::<Info>().metadata.buffer_len_index(buffer_id);
        load_static_info(ctx, rewriter, info_st, field, ty, buf_offs)
    }
}

#[op_interface_impl]
impl LowerInfoOp for ShapeOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter, info_st: Value) -> Value {
        let dim = self.dim(ctx);
        let field = static_meta_field(ctx);
        let ty = self.result_type(ctx);
        let buffer_id = self.buffer_idx(ctx).0;
        let buf_offs = ctx.aux_ty::<Info>().metadata.shape_offset_index(buffer_id);
        let dyn_offs = load_static_info(ctx, rewriter, info_st, field, ty, buf_offs);
        load_dyn_meta(ctx, rewriter, info_st, self.result_type(ctx), dyn_offs, dim)
    }
}

#[op_interface_impl]
impl LowerInfoOp for StrideOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter, info_st: Value) -> Value {
        let dim = self.dim(ctx);
        let field = static_meta_field(ctx);
        let ty = self.result_type(ctx);
        let buffer_id = self.buffer_idx(ctx).0;
        let buf_offs = ctx.aux_ty::<Info>().metadata.stride_offset_index(buffer_id);
        let dyn_offs = load_static_info(ctx, rewriter, info_st, field, ty, buf_offs);
        load_dyn_meta(ctx, rewriter, info_st, self.result_type(ctx), dyn_offs, dim)
    }
}

fn load_static_info(
    ctx: &mut Context,
    rewriter: &mut MatchRewriter,
    info_st: Value,
    field: usize,
    ty: TypeHandle,
    offset: usize,
) -> Value {
    let scope = Scope::from_context_and_inserter(ctx, rewriter);
    let buf_offs = const_int32(&scope, offset as u32);
    let static_offs = const_int32(&scope, field as u32);
    let out_ty = ty_to_spirv_dialect(ctx, ty);
    let align = out_ty.align(ctx) as u32;
    let ptr = info_ptr(ctx, out_ty);

    let chain = InBoundsAccessChainOp::new(ctx, ptr, info_st, vec![static_offs, buf_offs]);
    rewriter.append_op(ctx, &chain);
    let ptr = chain.get_result(ctx);
    let load = LoadOp::new(ctx, out_ty, ptr, MemoryAccess::ALIGNED, Some(align));
    rewriter.append_op(ctx, &load);
    load.get_result(ctx)
}

fn load_dyn_meta(
    ctx: &mut Context,
    rewriter: &mut MatchRewriter,
    info_st: Value,
    ty: TypeHandle,
    offset: Value,
    dim: Value,
) -> Value {
    let scope = Scope::from_context_and_inserter(ctx, rewriter);
    let dim_offs = scope.register_with_result(&IAddOp::new(ctx, offset, dim));
    let dyn_offs = const_int32(&scope, static_meta_field(ctx) as u32 + 1);
    let out_ty = ty_to_spirv_dialect(ctx, ty);
    let align = out_ty.align(ctx) as u32;
    let ptr = info_ptr(ctx, out_ty);

    let chain = InBoundsAccessChainOp::new(ctx, ptr, info_st, vec![dyn_offs, dim_offs]);
    rewriter.append_op(ctx, &chain);
    let ptr = chain.get_result(ctx);
    let load = LoadOp::new(ctx, out_ty, ptr, MemoryAccess::ALIGNED, Some(align));
    rewriter.append_op(ctx, &load);
    load.get_result(ctx)
}

fn static_meta_field(ctx: &Context) -> usize {
    ctx.aux_ty::<Info>().scalars.len()
}

fn info_ptr(ctx: &Context, ty: impl Into<TypeHandle>) -> TypeHandle {
    PointerType::get(ctx, ty.into(), StorageClass::PhysicalStorageBuffer).into()
}
