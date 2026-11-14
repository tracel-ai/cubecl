pub mod manual;
pub mod tensor;

pub use manual::*;

use core::cell::Ref;

use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        interfaces::TypedExt,
        prelude::*,
    },
};

use crate::{
    metal::{metal_op, ty::metal_ty},
    shared::{
        CompilationOptions, CppValue, DeclareMatrixOp,
        ty::{TypeExtCPP, TypeToCPP},
        wmma_api_base::{self, as_scalar_ptr},
    },
    target::Metal,
};

// 8x8x8 fragments are simdgroup matrices, any other shape a tensor operation's.
metal_ty!(MatrixType, |ty, ctx| {
    if tensor::is_tensor_op(&ty.shape) {
        return tensor::fragment_ty(ctx, ty);
    }
    format!("simdgroup_{}8x8", ty.elem_ty.to_cpp(ctx))
});

/// Whether `matrix` is a tensor operation's fragment.
fn tensor_op(ctx: &Context, matrix: impl Typed) -> bool {
    tensor::is_tensor_op(&matrix_ty(ctx, matrix).shape)
}

metal_op!(DeclareMatrixOp, |op, ctx| {
    if tensor_op(ctx, op.get_result(ctx)) {
        return tensor::declare(ctx, op);
    }
    wmma_api_base::compile_matrix_declaration(
        ctx,
        op.get_result(ctx),
        op.value_ty(ctx).get_type(ctx),
    )
});

metal_op!(FillOp, |op, ctx| {
    if tensor_op(ctx, op.matrix(ctx)) {
        return tensor::fill(ctx, op);
    }
    let mat = op.matrix(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let ty = op.value(ctx).get_type(ctx).to_cpp(ctx);

    format!("*{mat} = make_filled_simdgroup_matrix<{ty}, 8, 8>({value});\n",)
});

metal_op!(LoadOp, |op, ctx| {
    if tensor_op(ctx, op.matrix(ctx)) {
        return tensor::load(ctx, op);
    }
    let frag = op.matrix(ctx).name(ctx);
    let ptr = as_scalar_ptr(ctx, op.source(ctx));
    let stride = op.stride(ctx).name(ctx);
    // As on CUDA: A and B load in their own layout, an accumulator in the load's.
    let mat_ty = matrix_ty(ctx, op.matrix(ctx));
    let layout = match mat_ty.ident {
        MatrixIdent::A | MatrixIdent::B => mat_ty.layout,
        MatrixIdent::Accumulator => op.layout(ctx).0,
    };
    let transpose = matches!(layout, MatrixLayout::ColMajor);
    format!("simdgroup_load(*{frag}, {ptr}, {stride}, 0, {transpose});\n")
});

metal_op!(StoreOp, |op, ctx| {
    if tensor_op(ctx, op.matrix(ctx)) {
        return tensor::store(ctx, op);
    }
    let mat = op.matrix(ctx).name(ctx);
    let destination = as_scalar_ptr(ctx, op.destination(ctx));
    let stride = op.stride(ctx).name(ctx);
    let transpose = matches!(op.layout(ctx).0, MatrixLayout::ColMajor);
    format!(
        "
simdgroup_store(*{mat}, {destination}, {stride}, 0, {transpose});
simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
});

metal_op!(MultiplyAccumulateOp, |op, ctx| {
    if tensor_op(ctx, op.mat_a(ctx)) {
        return tensor::execute(ctx, op);
    }
    let a = op.mat_a(ctx).name(ctx);
    let b = op.mat_b(ctx).name(ctx);
    let c = op.mat_c(ctx).name(ctx);
    let d = op.mat_d(ctx).name(ctx);
    format!("simdgroup_multiply_accumulate(*{d}, *{a}, *{b}, *{c});\n")
});

metal_op!(CastOp, |op, ctx| {
    if tensor_op(ctx, op.output(ctx)) {
        return tensor::cast(ctx, op);
    }
    let input = op.input(ctx).name(ctx);
    let output = op.output(ctx).name(ctx);
    let output_ty = matrix_ty(ctx, op.output(ctx));
    let ty = output_ty.elem_ty.to_cpp(ctx);
    let threads_per_simdgroup = ctx.aux_ty::<CompilationOptions>().warp_size;
    let elements_held_by_each_thread =
        elements_held_by_each_thread(&output_ty.shape, threads_per_simdgroup);
    format!(
        "
simdgroup_barrier(mem_flags::mem_none);
for(int e=0; e<{elements_held_by_each_thread}; e++) {{
    {output}->thread_elements()[e] = {ty}({input}->thread_elements()[e]);
}}"
    )
});

fn elements_held_by_each_thread(shape: &MatrixShape, threads_per_simdgroup: usize) -> usize {
    (shape.m * shape.n) / threads_per_simdgroup
}

fn matrix_ty(ctx: &Context, ty: impl Typed) -> Ref<'_, MatrixType> {
    let ty = ty.unwrap_ptr(ctx).deref(ctx);
    Ref::map(ty, |ty| {
        ty.downcast_ref::<MatrixType>().expect("Should be matrix")
    })
}
