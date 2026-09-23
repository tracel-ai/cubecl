//! AMDGPU plane operations.

use crate::{
    amdgpu::intrinsic::lane_id_ops,
    prelude::*,
    shared::plane::{PlaneLowering, call_intrinsic, mask_ty, route_words, shl},
};

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

        route_words(ctx, rw, value, value_ty, |ctx, rw, word| {
            call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, word])
        })
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
