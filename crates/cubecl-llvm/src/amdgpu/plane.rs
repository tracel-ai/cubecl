//! AMDGPU plane operations.

use crate::amdgpu::intrinsic::lane_id_ops;
use crate::shared::intrinsic::i32_ty;
use crate::shared::plane::{
    PlaneLowering, bitcast, call_intrinsic, mask_ty, narrow_from_i32, shl, widen_to_i32,
};
use crate::shared::to_llvm::prelude::*;

/// Lane routing uses byte addresses.
const DS_BPERMUTE: &str = "llvm.amdgcn.ds.bpermute";

const BALLOT: &str = "llvm.amdgcn.ballot";

pub struct AmdGpuPlane;

impl PlaneLowering for AmdGpuPlane {
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let (ops, lane) = lane_id_ops(ctx);
        for op in ops {
            rw.insert_operation(ctx, op);
        }
        lane
    }

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

        if words == 1 {
            let as_i32 = widen_to_i32(ctx, rw, value, bits);
            let routed = call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, as_i32]);
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
        let name = format!("{BALLOT}.{}", llvm_mangled_ty(ctx, ty));
        call_intrinsic(ctx, rw, &name, ty, vec![predicate])
    }
}

pub(crate) fn lane_id(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    AmdGpuPlane.lane_id(ctx, rw)
}
