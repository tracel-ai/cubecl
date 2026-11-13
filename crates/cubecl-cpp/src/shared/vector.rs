use cubecl_core::{
    self as cubecl, define_scalar, define_size,
    ir::{
        FastMath,
        dialect::vector::*,
        interfaces::{TypedExt, fast_math::FastMathOp},
        prelude::*,
    },
    prelude::*,
};
use itertools::Itertools;

use crate::{
    shared::{
        binary::lower_binop,
        lowering::LowerOp,
        scoped_block, shared_op_with_out,
        ty::{TypeExtCPP, TypedExtCPP},
        unary::lower_unop,
    },
    target::{CtxTarget, Target},
};

shared_op_with_out!(CompositeConstructOp, |op, ctx| {
    let values = op.values(ctx).iter().map(|it| it.name(ctx)).join(", ");
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("{ty}{{{values}}}")
});

shared_op_with_out!(VectorBroadcastOp, |op, ctx| {
    let vec = op.get_result(ctx).vector_size(ctx);
    let values = (0..vec).map(|_| op.input(ctx).name(ctx)).join(", ");
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("{ty}{{{values}}}")
});

shared_op_with_out!(CompositeInsertOp, |op, ctx| {
    assert!(op.composite(ctx).is_vector(ctx));
    let vector_size = op.composite(ctx).vector_size(ctx);
    let vector = op.composite(ctx).name(ctx);
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

shared_op_with_out!(CompositeExtractOp, |op, ctx| {
    assert!(op.composite(ctx).is_vector(ctx));
    let vector = op.composite(ctx).name(ctx);
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
    format!("reinterpret_cast<const {elem_ty}*>(&{vector})[{index}]")
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

/// Starting from `0`, as [`sum`] does, turns a sum of `-0`s into `+0`.
#[cube]
fn signed_zero_sum<T: Float, N: Size>(input: Vector<T, N>) -> T {
    let mut sum = input.extract(0usize);
    #[unroll]
    for i in 1..input.vector_size() {
        sum += input.extract(i);
    }
    sum
}

#[cube]
fn dot<T: Float, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> T {
    sum(lhs * rhs)
}

#[cube]
fn signed_zero_dot<T: Float, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> T {
    signed_zero_sum(lhs * rhs)
}

fn keeps_zero_sign(op: &impl FastMathOp, ctx: &Context) -> bool {
    op.fast_math(ctx)
        .is_some_and(|fast_math| !fast_math.contains(FastMath::UnsignedZero))
}

// MSL's `length`, `normalize` and `dot` cover `half` and `float` vectors, not `bfloat` ones.
lower_unop!(MagnitudeOp, magnitude, |op, ctx| {
    ctx.target() != Target::Metal || op.input(ctx).is_bf16(ctx)
});

lower_unop!(NormalizeOp, normalize, |op, ctx| {
    ctx.target() != Target::Metal || op.input(ctx).is_bf16(ctx)
});

lower_unop!(ISumOp, sum);
lower_binop!(SDotOp, dot);
lower_binop!(UDotOp, dot);

#[op_interface_impl]
impl LowerOp for FSumOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let input = self.input(scope.ctx());
        scope.register_value_type::<T, S>(input);
        let sum = match keeps_zero_sign(self, scope.ctx()) {
            true => signed_zero_sum::expand::<T, S>(scope, input.into()),
            false => sum::expand::<T, S>(scope, input.into()),
        };
        vec![sum.read_value(scope)]
    }
}

#[op_interface_impl]
impl LowerOp for FDotOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        ctx.target() != Target::Metal || self.lhs(ctx).is_bf16(ctx)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let (lhs, rhs) = (self.lhs(scope.ctx()), self.rhs(scope.ctx()));
        scope.register_value_type::<T, S>(rhs);
        let dot = match keeps_zero_sign(self, scope.ctx()) {
            true => signed_zero_dot::expand::<T, S>(scope, lhs.into(), rhs.into()),
            false => dot::expand::<T, S>(scope, lhs.into(), rhs.into()),
        };
        vec![dot.read_value(scope)]
    }
}
