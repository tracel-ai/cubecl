//! AMDGPU plane operations.

use crate::{
    amdgpu::intrinsic::lane_id_ops,
    prelude::*,
    shared::plane::{PlaneLowering, and, icmp, mask_ty, route_words, select, shl, xor},
};

/// Lane routing uses byte addresses.
const DS_BPERMUTE: &str = "llvm.amdgcn.ds.bpermute";

const BALLOT: &str = "llvm.amdgcn.ballot";

/// Swaps the two 32-lane halves of a wave64 (gfx11 and later).
const PERMLANE64: &str = "llvm.amdgcn.permlane64.i32";

/// Whether a wave64 here runs as two 32-lane halves that `ds_bpermute` does not cross: RDNA
/// (gfx10 and later) compiled for wave64. A lane reading the other half then reads the halves
/// swapped by `permlane64`, as ACO does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalvedWave(pub bool);

impl CtxHalvedWave for Context {}

pub trait CtxHalvedWave: ContextExt {
    fn halved_wave(&self) -> bool {
        self.aux_ty::<HalvedWave>().0
    }
    fn set_halved_wave(&mut self, halved: bool) {
        self.set_aux_ty(HalvedWave(halved));
    }
}

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

        if !ctx.halved_wave() {
            return route_words(ctx, rw, value, value_ty, |ctx, rw, word| {
                call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, word])
            });
        }

        // The source lies in the other half where its lane differs from this one in bit 5. That
        // half is read from the halves swapped, at the source's lane in the swapped wave: the
        // hardware routes `ds_bpermute` by the lane's index within the half, while LLVM folds a
        // `ds_bpermute` whose lanes it can prove into DPP or `permlane64` as if it routed across
        // the whole wave, and both read the source there.
        let lane = self.lane_id(ctx, rw);
        let differs = xor(ctx, rw, lane, src_lane);
        let half = insert_i32_const(ctx, rw, 32);
        let half_bit = and(ctx, rw, differs, half);
        let zero = insert_i32_const(ctx, rw, 0);
        let crosses = icmp(ctx, rw, ICmpPredicateAttr::NE, half_bit, zero);
        let swapped_lane = xor(ctx, rw, src_lane, half);
        let swapped_addr = shl(ctx, rw, swapped_lane, two);
        route_words(ctx, rw, value, value_ty, |ctx, rw, word| {
            let same = call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, word]);
            let swapped = call_intrinsic(ctx, rw, PERMLANE64, i32_ty, vec![word]);
            let other = call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![swapped_addr, swapped]);
            select(ctx, rw, crosses, other, same)
        })
    }

    /// Where the wave runs as halves and the mask is known, the half it reads is known too, and
    /// the shuffle is one `ds_bpermute`, of the halves swapped where the mask crosses them.
    fn shuffle_xor(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        mask: Value,
        value_ty: TypeHandle,
    ) -> Value {
        let lane = self.lane_id(ctx, rw);
        let src = xor(ctx, rw, lane, mask);
        let known = match ctx.halved_wave() {
            true => constant_u32(ctx, mask),
            false => None,
        };
        let Some(bits) = known else {
            return self.shuffle(ctx, rw, value, src, value_ty);
        };

        let i32_ty = i32_ty(ctx);
        let two = insert_i32_const(ctx, rw, 2);
        let crosses = bits & 32 != 0;
        let read_lane = match crosses {
            true => {
                let half = insert_i32_const(ctx, rw, 32);
                xor(ctx, rw, src, half)
            }
            false => src,
        };
        let addr = shl(ctx, rw, read_lane, two);
        route_words(ctx, rw, value, value_ty, |ctx, rw, word| {
            let read = match crosses {
                true => call_intrinsic(ctx, rw, PERMLANE64, i32_ty, vec![word]),
                false => word,
            };
            call_intrinsic(ctx, rw, DS_BPERMUTE, i32_ty, vec![addr, read])
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

/// The value a constant defines, where `value` is an integer constant.
fn constant_u32(ctx: &Context, value: Value) -> Option<u32> {
    let def = value.defining_op()?;
    let constant = Operation::get_op::<ConstantOp>(def, ctx)?;
    let attr = constant.get_attr_builtin_constant_value(ctx)?;
    attr.downcast_ref::<IntegerAttr>()
        .map(|int| int.value().to_u64() as u32)
}
