use cubecl_core::ir::{dialect::vector::*, interfaces::TypedExt, prelude::*};

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

metal_op_with_out!(NormalizeOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let scalar_ty = op.result_type(ctx).to_cpp(ctx);
    let vec = op.input(ctx).vector_size(ctx);
    let input = format!("reinterpret_cast<const thread {scalar_ty}{vec}&>({input})");
    format!("normalize({input})")
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
metal_op_with_out!(VectorInsertOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).0;
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{vector_ty} tmp = {vector};")
        format!("tmp.i_{index} = {value};")
        "return tmp;"
    )
});

metal_op_with_out!(VectorInsertDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.value(ctx).get_type(ctx).to_cpp(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{vector_ty} tmp = {vector};")
        format!("*(reinterpret_cast<thread {elem_ty}*>(&tmp) + {index}) = {value};")
        "return tmp;"
    )
});

metal_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("reinterpret_cast<thread {elem_ty}*>({vector})[{index}]")
});
