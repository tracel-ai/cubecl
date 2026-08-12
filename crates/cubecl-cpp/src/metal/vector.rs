use cubecl_core::ir::{dialect::vector::*, interfaces::TypedExt, prelude::*};

use crate::{
    metal::{metal_op, metal_op_with_out},
    shared::{CppValue, ty::TypeExtCPP},
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

// Copy-then-assign rather than an aggregate literal, working around a Metal compiler bug that
// makes long chains of literal constructors blow up combinatorially. Emitted as statements
// because the result must be assigned field-wise and MSL has no lambdas.
metal_op!(CompositeInsertOp, |op, ctx| {
    assert!(op.composite(ctx).is_vector(ctx));
    let vector = op.composite(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).0;
    let out = op.get_result(ctx).name(ctx);
    let vector_ty = op.composite(ctx).get_type(ctx).to_cpp(ctx);
    format!("{vector_ty} {out} = {vector};\n{out}.i_{index} = {value};\n")
});

metal_op!(VectorInsertDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.value(ctx).get_type(ctx).to_cpp(ctx);
    let out = op.get_result(ctx).name(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    format!(
        "{vector_ty} {out} = {vector};\n\
         *(reinterpret_cast<thread {elem_ty}*>(&{out}) + {index}) = {value};\n"
    )
});

// Indexes the vector's storage, so it casts the ADDRESS of the vector, not its value.
metal_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("reinterpret_cast<const thread {elem_ty}*>(&{vector})[{index}]")
});
