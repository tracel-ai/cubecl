use cubecl_core::{
    self as cubecl,
    ir::{dialect::math::TanhOp, prelude::*},
    prelude::*,
};

use crate::{shared::ty::TypedExtCPP, shared::unary::lower_target_unop, target::Hip};

// HIP's fp16 headers don't provide `htanh`/`h2tanh` like CUDA does, so half precision `tanh` has
// to be computed in f32.
lower_target_unop!(TanhOp, tanh_via_f32, Hip, |op, ctx| op
    .get_result(ctx)
    .is_half(ctx));

#[cube]
fn tanh_via_f32<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::cast_from(Vector::<f32, N>::cast_from(input).tanh())
}
