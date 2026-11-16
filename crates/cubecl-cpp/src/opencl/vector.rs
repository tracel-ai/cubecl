use cubecl_core::ir::{dialect::vector::*, interfaces::TypedExt, prelude::*};
use itertools::Itertools;
use pliron::value::Value;

use crate::{
    opencl::opencl_op_with_out,
    shared::{CppValue, ty::TypeExtCPP},
};

/// Whether `value` is one of `OpenCL`'s own vectors (components `.s0`..), rather than a struct (`.i_0`..).
fn native(ctx: &Context, value: Value) -> bool {
    let ty = value.get_type(ctx).deref(ctx);
    ty.downcast_ref::<cubecl_core::ir::types::VectorType>()
        .is_some_and(|vector| crate::opencl::ty::native_vector(ctx, vector).is_some())
}

fn component(ctx: &Context, vector: Value, index: usize) -> String {
    match native(ctx, vector) {
        true => format!("{}.s{index:x}", vector.name(ctx)),
        false => format!("{}.i_{index}", vector.name(ctx)),
    }
}

// Vector literals and compound literals where the shared code has C++ brace construction.
opencl_op_with_out!(CompositeConstructOp, |op, ctx| {
    let values = op.values(ctx).iter().map(|it| it.name(ctx)).join(", ");
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    match native(ctx, op.get_result(ctx)) {
        true => format!("({ty})({values})"),
        false => format!("({ty}){{{values}}}"),
    }
});

opencl_op_with_out!(VectorBroadcastOp, |op, ctx| {
    let vec = op.get_result(ctx).vector_size(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    match native(ctx, op.get_result(ctx)) {
        true => format!("({ty})({})", op.input(ctx).name(ctx)),
        false => {
            let values = (0..vec).map(|_| op.input(ctx).name(ctx)).join(", ");
            format!("({ty}){{{values}}}")
        }
    }
});

opencl_op_with_out!(CompositeExtractOp, |op, ctx| {
    component(ctx, op.composite(ctx), op.index(ctx).0)
});

opencl_op_with_out!(CompositeInsertOp, |op, ctx| {
    let composite = op.composite(ctx);
    let size = composite.vector_size(ctx);
    let index = op.index(ctx).0;
    let ty = composite.get_type(ctx).to_cpp(ctx);
    let values = (0..size)
        .map(|i| match i == index {
            true => op.value(ctx).name(ctx).to_string(),
            false => component(ctx, composite, i),
        })
        .join(", ");
    match native(ctx, composite) {
        true => format!("({ty})({values})"),
        false => format!("({ty}){{{values}}}"),
    }
});

opencl_op_with_out!(VectorInsertDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.value(ctx).get_type(ctx).to_cpp(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    format!(
        "({{ {vector_ty} tmp = {vector}; ((__private {elem_ty}*)&tmp)[{index}] = {value}; tmp; }})"
    )
});

opencl_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    format!("({{ {vector_ty} tmp = {vector}; ((__private {elem_ty}*)&tmp)[{index}]; }})")
});
