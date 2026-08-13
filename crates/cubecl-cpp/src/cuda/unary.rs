use cubecl_core::ir::{ContextExt, dialect::math::TanhOp, prelude::*};

use crate::{
    shared::CompilationOptions,
    shared::ty::TypedExtCPP,
    shared::unary::{lower_target_unop, tanh_via_f32},
    target::Cuda,
};

// Half `tanh` emits `htanh`/`h2tanh`, which CUDA only declares from 12.8 on — below that the
// kernel fails to compile with `identifier "htanh" is undefined`. `supports_features.fast_tanh`
// already carries that version check, so fall back to computing in f32 when it is off.
lower_target_unop!(TanhOp, tanh_via_f32, Cuda, |op, ctx| {
    op.get_result(ctx).is_half(ctx)
        && !ctx.aux_ty::<CompilationOptions>().supports_features.fast_tanh
});
