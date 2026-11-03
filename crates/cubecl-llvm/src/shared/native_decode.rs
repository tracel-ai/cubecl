//! Where a minifloat cast is lowered: the software codec, or the target's own conversion.

use crate::prelude::*;
use cubecl_core::{
    ir::{
        NamedRewrite,
        dialect::general::CastOp,
        types::{
            Fp8Format,
            scalar::{Float4E2M1x2Type, Float8E4M3Type, Float8E5M2Type},
        },
    },
    post_processing::{
        fp4::{LowerFp4Cast, LowerFp4CastPass},
        minifloat::{Fp8Container, LowerMinifloatCast, LowerMinifloatCastPass},
    },
};

/// A minifloat a target decodes into floats itself: its casts to floats are left to the target's
/// `to_llvm` lowering, a pair of values an instruction. Every other cast from or to a minifloat,
/// encoding among them, is the software codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeDecode {
    E2M1x2,
    E4M3,
    E5M2,
}

impl NativeDecode {
    /// The format `ty`'s scalar is, where it is one of these.
    pub(crate) fn of_type(ctx: &Context, ty: TypeHandle) -> Option<Self> {
        let scalar = ty.scalar_ty(ctx);
        let scalar = scalar.deref(ctx);
        if scalar.is::<Float4E2M1x2Type>() {
            Some(NativeDecode::E2M1x2)
        } else if scalar.is::<Float8E4M3Type>() {
            Some(NativeDecode::E4M3)
        } else if scalar.is::<Float8E5M2Type>() {
            Some(NativeDecode::E5M2)
        } else {
            None
        }
    }

    /// The format a cast from `input` to `result` decodes, where it is a decode: from one of these
    /// formats to a float that is not a minifloat itself. A cast to a `bool` or an integer, or
    /// between minifloats, is the software codec's whichever the target.
    pub(crate) fn decoded_by(ctx: &Context, input: TypeHandle, result: TypeHandle) -> Option<Self> {
        let to_float = result.scalar_ty(ctx).is_float(ctx)
            && Self::of_type(ctx, result).is_none()
            && Fp8Format::of_type(ctx, result.scalar_ty(ctx)).is_none();
        Self::of_type(ctx, input).filter(|_| to_float)
    }

    /// Adds the passes lowering the minifloat casts a target decoding `native` itself does not.
    pub(crate) fn add_codec_passes(native: &'static [Self], passes: &mut OpPass<FuncOp, Passes>) {
        let fp8 = LowerMinifloatCast::new(Default::default(), Fp8Container::Bytes);
        let fp8_native = [NativeDecode::E4M3, NativeDecode::E5M2]
            .iter()
            .any(|format| native.contains(format));
        match fp8_native {
            false => passes.add_pass(LowerMinifloatCastPass::new(fp8)),
            true => passes.add_pass(MatchRewritePass::new(LowerSoftwareMinifloatCast {
                codec: fp8,
                native,
            })),
        }
        let fp4 = LowerFp4Cast::new(true, Fp8Container::Bytes);
        match native.contains(&NativeDecode::E2M1x2) {
            true => passes.add_pass(MatchRewritePass::new(LowerSoftwareFp4Cast(fp4))),
            false => passes.add_pass(LowerFp4CastPass::new(fp4)),
        }
    }
}

/// The fp8 software codec for the casts a target decoding `native` itself leaves to it: every cast
/// to an fp8 format, and every cast from one it does not decode.
#[derive(Clone, Copy, Debug, NamedRewrite)]
struct LowerSoftwareMinifloatCast {
    codec: LowerMinifloatCast,
    native: &'static [NativeDecode],
}

impl MatchRewrite for LowerSoftwareMinifloatCast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        let fp8 = |ty: TypeHandle| Fp8Format::of_type(ctx, ty.scalar_ty(ctx)).is_some();
        let (input, result) = (
            op.operand(ctx, 0).get_type(ctx),
            op.result(ctx).get_type(ctx),
        );
        let decoded = NativeDecode::decoded_by(ctx, input, result)
            .is_some_and(|format| self.native.contains(&format));
        (fp8(input) && !decoded) || fp8(result)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        self.codec.rewrite(ctx, rewriter, op)
    }
}

/// The fp4 software codec for every cast from or to `e2m1x2` but a decode, on a target that
/// decodes it itself.
#[derive(Clone, Copy, Debug, NamedRewrite)]
struct LowerSoftwareFp4Cast(LowerFp4Cast);

impl MatchRewrite for LowerSoftwareFp4Cast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        let (input, result) = (
            op.operand(ctx, 0).get_type(ctx),
            op.result(ctx).get_type(ctx),
        );
        let decoded = NativeDecode::decoded_by(ctx, input, result) == Some(NativeDecode::E2M1x2);
        !decoded && self.0.r#match(ctx, op)
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
