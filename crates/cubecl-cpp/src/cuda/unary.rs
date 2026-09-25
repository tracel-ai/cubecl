use cubecl_core::ir::{dialect::math::TanhOp, prelude::*};

use crate::{
    shared::CompilationOptions,
    shared::ty::TypedExtCPP,
    shared::unary::{lower_target_unop, tanh_via_f32},
    target::Cuda,
};

// CUDA headers before 12.8 lack htanh and h2tanh. Lower before packing.
lower_target_unop!(TanhOp, tanh_via_f32, Cuda, |op, ctx| {
    op.get_result(ctx).is_half(ctx)
        && !ctx
            .aux_ty::<CompilationOptions>()
            .supports_features
            .fast_tanh
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        shared::lowering::LowerOpsCpp,
        target::{CtxTarget, Target},
    };
    use cubecl_core::ir::{ConstantValue, ElemType, FloatKind};
    use pliron::{attribute::boxed_attr_cast, builtin::ops::ConstantOp, op::Op};

    #[test]
    fn half_tanh_uses_f32_when_native_cuda_tanh_is_unavailable() {
        for (kind, fast_tanh, expected) in [
            (FloatKind::F16, false, true),
            (FloatKind::BF16, false, true),
            (FloatKind::F16, true, false),
            (FloatKind::BF16, true, false),
            (FloatKind::F32, false, false),
        ] {
            let mut ctx = Context::new();
            ctx.set_target(Target::Cuda);
            let mut options = CompilationOptions::default();
            options.supports_features.fast_tanh = fast_tanh;
            ctx.set_aux_ty(options);
            let ty = ElemType::Float(kind);
            let value = boxed_attr_cast(ConstantValue::Float(1.0).as_attribute(&ctx, ty)).unwrap();
            let input = ConstantOp::new(&mut ctx, value).get_result(&ctx);
            let op = TanhOp::new(&mut ctx, input);
            assert_eq!(
                LowerOpsCpp::<Cuda>::default().r#match(&ctx, op.get_operation()),
                expected,
                "{kind:?}, fast_tanh={fast_tanh}"
            );
        }
    }
}
