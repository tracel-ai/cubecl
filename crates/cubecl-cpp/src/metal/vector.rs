use cubecl_core::ir::{dialect::vector::*, interfaces::TypedExt, prelude::*};
use itertools::Itertools;

use crate::{
    metal::metal_op_with_out,
    shared::{scoped_block, ty::TypeExtCPP},
};

metal_op_with_out!(MagnitudeOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let scalar_ty = op.result_type(ctx).to_cpp(ctx);
    let vec = op.input(ctx).vector_size(ctx);
    let input = format!("reinterpret_cast<const thread {scalar_ty}{vec}&>({input})");
    format!("length({input})")
});

// The result is a vector too: the MSL vector is named from the scalar, and the value MSL returns
// is reinterpreted back into the vector struct the result is declared as.
metal_op_with_out!(NormalizeOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let scalar_ty = op.input(ctx).scalar_ty(ctx).to_cpp(ctx);
    let vec = op.input(ctx).vector_size(ctx);
    let msl_ty = format!("{scalar_ty}{vec}");
    let out_ty = op.result_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{msl_ty} normalized = normalize(reinterpret_cast<const thread {msl_ty}&>({input}));")
        format!("return reinterpret_cast<const thread {out_ty}&>(normalized);")
    )
});

metal_op_with_out!(FDotOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    let scalar_ty = op.result_type(ctx).to_cpp(ctx);
    let vec = op.lhs(ctx).vector_size(ctx);
    let reinterpret = format!("reinterpret_cast<const thread {scalar_ty}{vec}&>");
    format!("dot({reinterpret}({lhs}), {reinterpret}({rhs}))")
});

// Workaround for Metal compiler bug that causes combinatorial explosion with large aggregate
// literal chains. No idea what they're doing wrong but this works around it. Keep it Metal only
// because it's slightly less clean and less analyzable than the literal constructor.
metal_op_with_out!(CompositeInsertOp, |op, ctx| {
    assert!(op.composite(ctx).is_vector(ctx));
    let vector = op.composite(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).0;
    let vector_ty = op.composite(ctx).get_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{vector_ty} tmp = {vector};")
        format!("tmp.i_{index} = {value};")
        "return tmp;"
    )
});

// A vector is a struct of `i_0`, `i_1`, … fields, so an index known only at run time selects
// among them. Reached through a pointer instead, the vector would leave its registers for the
// thread's stack, which Apple GPUs keep in device memory: a store and a load at every access.
metal_op_with_out!(VectorInsertDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    let selects = (0..op.vector(ctx).vector_size(ctx))
        .map(|i| format!("tmp.i_{i} = {index} == {i}u ? {value} : tmp.i_{i};"))
        .join("\n");
    scoped_block!(
        format!("{vector_ty} tmp = {vector};")
        selects
        "return tmp;"
    )
});

metal_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let last = op.vector(ctx).vector_size(ctx) - 1;
    (0..last)
        .rev()
        .fold(format!("{vector}.i_{last}"), |rest, i| {
            format!("({index} == {i}u ? {vector}.i_{i} : {rest})")
        })
});
