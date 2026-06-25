use cubecl_core::{
    self as cubecl,
    ir::{
        dialect::vector::{
            DotOp, MagnitudeOp, NormalizeOp, SumOp, VectorBroadcastOp, VectorExtractDynamicOp,
            VectorExtractOp, VectorInitOp, VectorInsertDynamicOp, VectorInsertOp,
        },
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};
use itertools::Itertools;

use crate::{
    shared::{
        binary::lower_binop, scoped_block, shared_op_with_out, ty::TypeExtCPP, unary::lower_unop,
    },
    target::{CtxTarget, Target},
};

shared_op_with_out!(VectorInitOp, |op, ctx| {
    let values = op.values(ctx).iter().map(|it| it.name(ctx)).join(", ");
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("{ty}{{{values}}};")
});

shared_op_with_out!(VectorBroadcastOp, |op, ctx| {
    let vec = op.get_result(ctx).vector_size(ctx);
    let values = (0..vec).map(|_| op.input(ctx).name(ctx)).join(", ");
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("{ty}{{{values}}};")
});

shared_op_with_out!(VectorInsertOp, |op, ctx| {
    let vector_size = op.vector(ctx).vector_size(ctx);
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).0;
    let new_values = (0..vector_size)
        .map(|i| {
            if i == index {
                value.to_string()
            } else {
                format!("{vector}.i_{i}")
            }
        })
        .join(", ");
    format!("{{{new_values}}}")
});

shared_op_with_out!(VectorExtractOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).0;
    format!("{vector}.i_{index}")
});

shared_op_with_out!(VectorInsertDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.value(ctx).get_type(ctx).to_cpp(ctx);
    let vector_ty = op.vector(ctx).get_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{vector_ty} tmp = {vector};")
        format!("*(reinterpret_cast<{elem_ty}*>(&tmp) + {index}) = {value};")
        "return tmp;"
    )
});

shared_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let elem_ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("reinterpret_cast<{elem_ty}*>({vector})[{index}]")
});

#[cube]
fn magnitude<T: Float, N: Size>(input: Vector<T, N>) -> T {
    let mut out = T::from_int(0);
    #[unroll]
    for i in 0..input.vector_size() {
        let val = input.extract(i);
        out += val * val;
    }
    out.sqrt()
}

#[cube]
fn normalize<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    let mut norm = T::from_int(0);
    #[unroll]
    for i in 0..input.vector_size() {
        let val = input.extract(i);
        norm += val * val;
    }
    input * Vector::cast_from(norm.inverse_sqrt())
}

#[cube]
fn sum<T: Float, N: Size>(input: Vector<T, N>) -> T {
    let mut sum = T::from_int(0);
    #[unroll]
    for i in 0..input.vector_size() {
        sum += input.extract(i);
    }
    sum
}

#[cube]
fn dot<T: Float, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> T {
    sum(lhs * rhs)
}

lower_unop!(MagnitudeOp, magnitude, |_, ctx| {
    ctx.target() != Target::Metal
});

lower_unop!(NormalizeOp, normalize, |_, ctx| {
    ctx.target() != Target::Metal
});

lower_unop!(SumOp, sum);
lower_binop!(DotOp, dot, |_, ctx| ctx.target() != Target::Metal);
