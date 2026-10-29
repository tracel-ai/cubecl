//! Where an `e2m1x2` cast is lowered: the software codec, or the target's own conversion.

use crate::prelude::*;
use cubecl_core::{
    ir::{NamedRewrite, dialect::general::CastOp, types::scalar::Float4E2M1x2Type},
    post_processing::{
        fp4::{LowerFp4Cast, LowerFp4CastPass},
        minifloat::Fp8Container,
    },
};

/// How a target decodes `e2m1x2` into floats. Encoding is always the software codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4Decode {
    /// Through `f16` pairs, the software codec every target holds.
    Software,
    /// The cast is left to the target's `to_llvm` lowering, which converts a pair in one
    /// instruction.
    Native,
}

impl Fp4Decode {
    /// Adds the pass that lowers the `e2m1x2` casts this target does not convert itself.
    pub(crate) fn add_passes(self, passes: &mut OpPass<FuncOp, Passes>) {
        let codec = LowerFp4Cast::new(true, Fp8Container::Bytes);
        match self {
            Fp4Decode::Software => passes.add_pass(LowerFp4CastPass::new(codec)),
            Fp4Decode::Native => passes.add_pass(MatchRewritePass::new(LowerFp4Encode(codec))),
        }
    }
}

/// The software codec for the casts to `e2m1x2` only, on a target that decodes it itself.
#[derive(Clone, Copy, Debug, NamedRewrite)]
struct LowerFp4Encode(LowerFp4Cast);

impl MatchRewrite for LowerFp4Encode {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<CastOp>(ctx)
            && is_e2m1x2(ctx, op.result(ctx).get_type(ctx))
            && self.0.r#match(ctx, op)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        self.0.rewrite(ctx, rewriter, op)
    }
}

/// Whether `ty` holds `e2m1x2` pairs, as a scalar or a vector of them.
pub(crate) fn is_e2m1x2(ctx: &Context, ty: TypeHandle) -> bool {
    ty.scalar_ty(ctx).deref(ctx).is::<Float4E2M1x2Type>()
}
