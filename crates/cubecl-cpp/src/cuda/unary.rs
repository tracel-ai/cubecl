use cubecl_core::ir::{dialect::math::TanhOp, prelude::*};

use crate::{
    shared::CompilationOptions,
    shared::ty::TypedExtCPP,
    shared::unary::{lower_target_unop, tanh_via_f32},
    target::Cuda,
};

// `htanh`/`h2tanh` reach the hardware `tanh.approx.f16`, which is worth having
// over an f32 round trip — but the intrinsics only appear in the CUDA 12.8
// headers, and the instruction they lower to needs sm_75. `fast_tanh` carries
// both conditions. Where either is missing, half precision `tanh` is computed
// in f32 the way it always is on HIP.
//
// This runs before the packing pass, so a half pair that would have become
// `h2tanh` is already f32 arithmetic by the time packing looks at it.
lower_target_unop!(TanhOp, tanh_via_f32, Cuda, |op, ctx| {
    op.get_result(ctx).is_half(ctx)
        && !ctx
            .aux_ty::<CompilationOptions>()
            .supports_features
            .fast_tanh
});
