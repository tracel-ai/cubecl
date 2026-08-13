use cubecl_core::ir::{dialect::math::TanhOp, prelude::*};

use crate::{
    shared::ty::TypedExtCPP,
    shared::unary::{lower_target_unop, tanh_via_f32},
    target::Hip,
};

// HIP's fp16 headers don't provide `htanh`/`h2tanh`, so half precision `tanh` has to be
// computed in f32.
lower_target_unop!(TanhOp, tanh_via_f32, Hip, |op, ctx| op
    .get_result(ctx)
    .is_half(ctx));
