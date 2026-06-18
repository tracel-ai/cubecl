use core::cell::Ref;

use cubecl_core::{
    cmma::{MatrixLayout, MatrixShape, MatrixType},
    ir::{
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        prelude::*,
    },
};

use crate::{
    metal::{metal_op, ty::metal_ty},
    shared::{
        CppValue, DeclareMatrixOp,
        ty::{TypeExtCPP, TypeToCPP},
        wmma_api_base,
    },
    target::Metal,
};

pub fn compile_cmma_includes_metal() -> String {
    "#include <metal_simdgroup_matrix>\n".into()
}

metal_ty!(MatrixType, |ty, ctx| {
    let MatrixShape { m, n, k } = ty.shape;
    let ty = ty.elem_ty.to_cpp(ctx);
    // currently as of Metal 3.2 only fragments of 8x8x8 are supported
    if m != 8 || n != 8 || k != 8 {
        panic!("{m}x{n}x{k} fragments not supported. Only 8x8x8 fragments are supported.");
    }
    format!("simdgroup_{ty}8x8")
});

metal_op!(DeclareMatrixOp, |op, ctx| {
    wmma_api_base::compile_matrix_declaration(
        ctx,
        op.get_result(ctx),
        op.value_ty(ctx).get_type(ctx),
    )
});

metal_op!(FillOp, |op, ctx| {
    let mat = op.matrix(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let ty = op.value(ctx).get_type(ctx).to_cpp(ctx);

    format!("*{mat} = make_filled_simdgroup_matrix<{ty}, 8, 8>({value});",)
});

metal_op!(LoadOp, |op, ctx| {
    let frag = op.matrix(ctx).name(ctx);
    let ptr = op.source(ctx).name(ctx);
    let stride = op.stride(ctx).name(ctx);
    let mat_ty = matrix_ty(ctx, op.matrix(ctx));
    let transpose = match mat_ty.layout {
        MatrixLayout::RowMajor | MatrixLayout::Undefined => false,
        MatrixLayout::ColMajor => true,
    };
    format!("simdgroup_load(*{frag}, {ptr}, {stride}, 0, {transpose});")
});

metal_op!(StoreOp, |op, ctx| {
    let mat = op.matrix(ctx).name(ctx);
    let destination = op.destination(ctx).name(ctx);
    let stride = op.stride(ctx).name(ctx);
    format!(
        "
simdgroup_store({mat}, {destination}, {stride});
simdgroup_barrier(mem_flags::mem_none);"
    )
});

metal_op!(MultiplyAccumulateOp, |op, ctx| {
    let a = op.mat_a(ctx).name(ctx);
    let b = op.mat_b(ctx).name(ctx);
    let c = op.mat_c(ctx).name(ctx);
    let d = op.mat_d(ctx).name(ctx);
    format!("simdgroup_multiply_accumulate(*{d}, {a}, {b}, {c});")
});

metal_op!(CastOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let output = op.output(ctx).name(ctx);
    let ty = matrix_ty(ctx, op.output(ctx)).elem_ty.to_cpp(ctx);
    format!(
        "
simdgroup_barrier(mem_flags::mem_none);
for(int e=0; e<8; e++) {{
    {output}->thread_elements()[e] = {ty}({input}.thread_elements()[e]);
}}"
    )
});

fn matrix_ty(ctx: &Context, ty: impl Typed) -> Ref<'_, MatrixType> {
    let ty = ty.get_type(ctx).deref(ctx);
    Ref::map(ty, |ty| {
        ty.downcast_ref::<MatrixType>().expect("Should be matrix")
    })
}
