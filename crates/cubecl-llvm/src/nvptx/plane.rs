//! NVPTX plane operations.

use crate::{
    prelude::*,
    shared::plane::{PlaneLowering, call_intrinsic, route_words},
};

const LANEID: &str = "llvm.nvvm.read.ptx.sreg.laneid";

const SHFL_IDX: &str = "llvm.nvvm.shfl.sync.idx.i32";
const SHFL_UP: &str = "llvm.nvvm.shfl.sync.up.i32";
const SHFL_DOWN: &str = "llvm.nvvm.shfl.sync.down.i32";
const SHFL_BFLY: &str = "llvm.nvvm.shfl.sync.bfly.i32";

const VOTE_BALLOT: &str = "llvm.nvvm.vote.ballot.sync";
const VOTE_ALL: &str = "llvm.nvvm.vote.all.sync";
const VOTE_ANY: &str = "llvm.nvvm.vote.any.sync";

const ACTIVEMASK: &str = "llvm.nvvm.activemask";

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
        let mask = member_mask(ctx, rw);
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

/// The lanes a plane operation synchronizes: the ones executing it, as the C++ backend's
/// `__activemask()` does, so an operation inside a branch only a part of the plane takes waits
/// for that part rather than for lanes that never reach it.
fn member_mask(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    let ty = i32_ty(ctx);
    call_intrinsic(ctx, rw, ACTIVEMASK, ty, vec![])
}

fn vote(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    predicate: Value,
) -> Value {
    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let mask = member_mask(ctx, rw);
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
    let mask = member_mask(ctx, rw);
    let clamp = insert_i32_const(ctx, rw, clamp);

    route_words(ctx, rw, value, value_ty, |ctx, rw, word| {
        let args = vec![mask, word, operand, clamp];
        call_intrinsic(ctx, rw, name, i32_ty, args)
    })
}
