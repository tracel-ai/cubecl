//! What the warp's cross-lane instructions give the plane operations.
//!
//! Every routing here is `shfl.sync`, which moves 32 bits at a time, so wider and narrower
//! values are taken apart into words, routed, and put back together — the same shape as the
//! AMDGPU side's `ds_bpermute`.
//!
//! Unlike AMD, NVIDIA has an instruction for each direction rather than only for an absolute
//! lane, and a vote instruction for `all` and `any` rather than only a ballot. Those are taken
//! up as overrides of [`PlaneLowering`]'s defaults: the derivations in
//! [`shared::plane`](crate::shared::plane) would compute the same answers, at several ALU ops
//! and an extra shuffle each.

use crate::shared::intrinsic::i32_ty;
use crate::shared::plane::{PlaneLowering, bitcast, call_intrinsic, narrow_from_i32, widen_to_i32};
use crate::shared::to_llvm::prelude::*;

/// The lane's index within its warp.
const LANEID: &str = "llvm.nvvm.read.ptx.sreg.laneid";

/// `shfl.sync.<mode>.b32`, whose modes differ only in how the source lane is derived.
const SHFL_IDX: &str = "llvm.nvvm.shfl.sync.idx.i32";
const SHFL_UP: &str = "llvm.nvvm.shfl.sync.up.i32";
const SHFL_DOWN: &str = "llvm.nvvm.shfl.sync.down.i32";
const SHFL_BFLY: &str = "llvm.nvvm.shfl.sync.bfly.i32";

/// The vote instructions, which answer about the warp rather than routing within it.
const VOTE_BALLOT: &str = "llvm.nvvm.vote.ballot.sync";
const VOTE_ALL: &str = "llvm.nvvm.vote.all.sync";
const VOTE_ANY: &str = "llvm.nvvm.vote.any.sync";

/// Every lane of the warp takes part.
///
/// A cube operation is a whole-plane operation by definition: the frontend's `plane_*` are
/// documented as needing every unit of the plane to reach them, which is the same contract
/// `__shfl_sync(0xffffffff, ...)` is written under and what the C++ backend emits.
const FULL_MASK: i32 = -1;

/// The packed clamp and segment mask `shfl.sync` takes as its last operand: bits 4:0 bound the
/// source lane, bits 12:8 split the warp into segments.
///
/// A plane is the whole warp here, so the segment mask is zero throughout. The bound differs
/// by mode: `up` reads downwards and clamps at the bottom of the warp, everything else reads
/// upwards and clamps at the top. A lane whose source falls outside the bound keeps its own
/// value, which is exactly what `plane_shuffle_up` and `plane_shuffle_down` are specified to
/// do at the ends of the plane.
const CLAMP_TO_TOP: i32 = 0x1f;
const CLAMP_TO_BOTTOM: i32 = 0x00;

/// The NVPTX target's plane primitives.
pub struct NvptxPlane;

impl PlaneLowering for NvptxPlane {
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let ty = i32_ty(ctx);
        call_intrinsic(ctx, rw, LANEID, ty, vec![])
    }

    fn shuffle(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        src_lane: Value,
        value_ty: TypeHandle,
    ) -> Value {
        shfl(ctx, rw, SHFL_IDX, value, src_lane, CLAMP_TO_TOP, value_ty)
    }

    fn ballot_mask(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        predicate: Value,
    ) -> Value {
        let ty = i32_ty(ctx);
        let mask = insert_i32_const(ctx, rw, FULL_MASK);
        call_intrinsic(ctx, rw, VOTE_BALLOT, ty, vec![mask, predicate])
    }

    /// `shfl.sync.up` reads from `lane - delta` and leaves a lane with no source holding its
    /// own value, so the bounds check the default body builds is the hardware's already.
    fn shuffle_up(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        delta: Value,
        value_ty: TypeHandle,
    ) -> Value {
        shfl(ctx, rw, SHFL_UP, value, delta, CLAMP_TO_BOTTOM, value_ty)
    }

    /// The same downwards, clamped at the top of the warp instead.
    fn shuffle_down(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        delta: Value,
        value_ty: TypeHandle,
    ) -> Value {
        shfl(ctx, rw, SHFL_DOWN, value, delta, CLAMP_TO_TOP, value_ty)
    }

    /// `lane ^ mask` never leaves the warp, so the clamp never fires and this is a pure
    /// butterfly.
    fn shuffle_xor(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        mask: Value,
        value_ty: TypeHandle,
    ) -> Value {
        shfl(ctx, rw, SHFL_BFLY, value, mask, CLAMP_TO_TOP, value_ty)
    }

    /// One `vote.all.sync`, where the default body is a ballot compared against a second
    /// ballot of the running lanes.
    fn all(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        vote(ctx, rw, VOTE_ALL, input)
    }

    /// One `vote.any.sync`, where the default body is a ballot compared against zero.
    fn any(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        vote(ctx, rw, VOTE_ANY, input)
    }
}

/// Emits the vote instruction `name` over `predicate`, which answers `i1`.
fn vote(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    predicate: Value,
) -> Value {
    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let mask = insert_i32_const(ctx, rw, FULL_MASK);
    call_intrinsic(ctx, rw, name, bool_ty, vec![mask, predicate])
}

/// Routes `value` with the `shfl.sync` mode `name`, whose `operand` is a source lane for `idx`
/// and a delta or a butterfly mask for the others.
///
/// The instruction moves one 32 bit word, so anything else is bitcast to a whole number of
/// words, routed a word at a time, and bitcast back.
fn shfl(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    value: Value,
    operand: Value,
    clamp: i32,
    value_ty: TypeHandle,
) -> Value {
    let i32_ty = i32_ty(ctx);
    let mask = insert_i32_const(ctx, rw, FULL_MASK);
    let clamp = insert_i32_const(ctx, rw, clamp);

    let llvm_ty = cube_type_to_llvm(ctx, value_ty);
    let bits = value_ty.size_bits(ctx) as u32;
    let words = bits.div_ceil(32);

    // One word: route the bit pattern straight through.
    if words == 1 {
        let as_i32 = widen_to_i32(ctx, rw, value, bits);
        let args = vec![mask, as_i32, operand, clamp];
        let routed = call_intrinsic(ctx, rw, name, i32_ty, args);
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
        let args = vec![mask, word_value, operand, clamp];
        let one = call_intrinsic(ctx, rw, name, i32_ty, args);
        let op = llvm::InsertElementOp::new(ctx, acc, one, index);
        acc = insert(ctx, rw, &op);
    }
    bitcast(ctx, rw, acc, llvm_ty)
}
