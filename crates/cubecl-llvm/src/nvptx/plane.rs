//! NVPTX plane operations.

use crate::{
    prelude::*,
    shared::plane::{PlaneLowering, bitcast, call_intrinsic, narrow_from_i32, widen_to_i32},
};

const LANEID: &str = "llvm.nvvm.read.ptx.sreg.laneid";

const SHFL_IDX: &str = "llvm.nvvm.shfl.sync.idx.i32";
const SHFL_UP: &str = "llvm.nvvm.shfl.sync.up.i32";
const SHFL_DOWN: &str = "llvm.nvvm.shfl.sync.down.i32";
const SHFL_BFLY: &str = "llvm.nvvm.shfl.sync.bfly.i32";

const VOTE_BALLOT: &str = "llvm.nvvm.vote.ballot.sync";
const VOTE_ALL: &str = "llvm.nvvm.vote.all.sync";
const VOTE_ANY: &str = "llvm.nvvm.vote.any.sync";

/// Plane operations require participation from every lane.
const FULL_MASK: i32 = -1;

/// Shuffle bounds: bits 4:0 limit the source lane; bits 12:8 select the segment.
const CLAMP_TO_TOP: i32 = 0x1f;
const CLAMP_TO_BOTTOM: i32 = 0x00;

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

    fn all(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        vote(ctx, rw, VOTE_ALL, input)
    }

    fn any(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        vote(ctx, rw, VOTE_ANY, input)
    }
}

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

    if words == 1 {
        let as_i32 = widen_to_i32(ctx, rw, value, bits);
        let args = vec![mask, as_i32, operand, clamp];
        let routed = call_intrinsic(ctx, rw, name, i32_ty, args);
        return narrow_from_i32(ctx, rw, routed, bits, llvm_ty);
    }

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
