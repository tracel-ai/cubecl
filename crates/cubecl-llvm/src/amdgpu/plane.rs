//! What the wavefront's cross-lane instructions give the plane operations.
//!
//! Every routing here is `ds_bpermute`: each lane names the lane it wants to read from and the
//! hardware routes the value. It moves 32 bits at a time, so wider and narrower values are
//! taken apart into words, routed, and put back together.
//!
//! `ballot` is the exec mask of a predicate. The operations built on these two primitives —
//! `all`, `any`, `elect`, the directional shuffles — are in
//! [`shared::plane`](crate::shared::plane), which is where an AMDGPU-specific version of any
//! of them would go as an override.

use crate::amdgpu::intrinsic::lane_id_ops;
use crate::shared::intrinsic::i32_ty;
use crate::shared::plane::{
    PlaneLowering, bitcast, call_intrinsic, mask_ty, narrow_from_i32, shl, widen_to_i32,
};
use crate::shared::to_llvm::prelude::*;

/// Routes a 32-bit word between the lanes of a wavefront. The address is a byte address, so a
/// lane index has to be scaled by four.
const DS_BPERMUTE: &str = "llvm.amdgcn.ds.bpermute";

/// The exec mask of a predicate: one bit per lane of the wavefront, set where it holds.
const BALLOT: &str = "llvm.amdgcn.ballot";

/// The AMDGPU target's plane primitives.
pub struct AmdGpuPlane;

impl PlaneLowering for AmdGpuPlane {
    /// Recomputed rather than taken from the builtin, which has long since been substituted by
    /// the time the ops reach here. The optimizer folds the duplicates back together.
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let (ops, lane) = lane_id_ops(ctx);
        for op in ops {
            rw.insert_operation(ctx, op);
        }
        lane
    }

    /// `ds_bpermute` moves one word, so anything else is bitcast to a whole number of words,
    /// routed a word at a time, and bitcast back.
    fn shuffle(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        src_lane: Value,
        value_ty: TypeHandle,
    ) -> Value {
        let i32_ty = i32_ty(ctx);
        let two = insert_i32_const(ctx, rw, 2);
        let addr = shl(ctx, rw, src_lane, two);

        let llvm_ty = cube_type_to_llvm(ctx, value_ty);
        let bits = value_ty.size_bits(ctx) as u32;
        let words = bits.div_ceil(32);

        // One word: route the bit pattern straight through.
        if words == 1 {
            let as_i32 = widen_to_i32(ctx, rw, value, bits);
            let routed = call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, as_i32]);
            return narrow_from_i32(ctx, rw, routed, bits, llvm_ty);
        }

        // Several: through a vector of words, so each element is one routing.
        let words_ty = LlvmVectorType::get(ctx, i32_ty, words, VectorTypeKind::Fixed).into();
        let as_words = bitcast(ctx, rw, value, words_ty);

        let poison = llvm::PoisonOp::new(ctx, words_ty);
        let mut acc = insert(ctx, rw, &poison);
        for word in 0..words {
            let index = insert_i32_const(ctx, rw, word as i32);
            let extract = llvm::ExtractElementOp::new(ctx, as_words, index);
            let word_value = insert(ctx, rw, &extract);
            let one = call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, word_value]);
            let op = llvm::InsertElementOp::new(ctx, acc, one, index);
            acc = insert(ctx, rw, &op);
        }
        bitcast(ctx, rw, acc, llvm_ty)
    }

    fn ballot_mask(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        predicate: Value,
    ) -> Value {
        let ty = mask_ty(ctx);
        // The intrinsic is overloaded on its return type, so the mangled name carries the width.
        let name = format!("{BALLOT}.{}", llvm_mangled_ty(ctx, ty));
        call_intrinsic(ctx, rw, &name, ty, vec![predicate])
    }
}

/// This lane's index within its wavefront, for the lowerings outside this module that need it
/// — [`matrix`](super::matrix) indexes its fragments by it.
pub(crate) fn lane_id(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    AmdGpuPlane.lane_id(ctx, rw)
}
