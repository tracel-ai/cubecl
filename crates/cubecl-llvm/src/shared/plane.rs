//! Shared plane operations.

use crate::prelude::*;
use cubecl_core::ir::dialect::plane::{
    AllOp, AnyOp, BallotOp, BroadcastOp, ElectOp, ShuffleDownOp, ShuffleOp, ShuffleUpOp,
    ShuffleXorOp,
};

pub(crate) const CTTZ: &str = "llvm.cttz";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneDim(pub u32);

impl CtxPlaneDim for Context {}

pub trait CtxPlaneDim: ContextExt {
    fn plane_dim(&self) -> u32 {
        self.aux_ty::<PlaneDim>().0
    }
    fn set_plane_dim(&mut self, plane_dim: u32) {
        self.set_aux_ty(PlaneDim(plane_dim));
    }
}

#[derive(Debug, Error)]
#[error(
    "plane operations need a plane of more than one unit, which the CPU target does not have; \
     a runtime that lowers them here has advertised `Plane::Ops` it cannot honour"
)]
pub struct PlaneOpsUnsupported;

/// Target plane primitives with defaults for derived operations.
pub trait PlaneLowering {
    /// Lane index within the plane.
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value;

    /// Value from `src_lane`.
    fn shuffle(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        src_lane: Value,
        value_ty: TypeHandle,
    ) -> Value;

    /// Mask of lanes where the predicate holds.
    fn ballot_mask(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        predicate: Value,
    ) -> Value;

    /// Mask of active lanes.
    fn active_lanes(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let all = insert_bool_const(ctx, rw, true);
        self.ballot_mask(ctx, rw, all)
    }

    /// Value from the lane selected by XOR with `mask`.
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
        self.shuffle(ctx, rw, value, src, value_ty)
    }

    /// Value from a lower lane, or the current lane when out of bounds.
    fn shuffle_up(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        delta: Value,
        value_ty: TypeHandle,
    ) -> Value {
        let lane = self.lane_id(ctx, rw);
        let shifted = sub(ctx, rw, lane, delta);
        let has_source = icmp(ctx, rw, ICmpPredicateAttr::UGE, lane, delta);
        let src = select(ctx, rw, has_source, shifted, lane);
        self.shuffle(ctx, rw, value, src, value_ty)
    }

    /// Value from a higher lane, or the current lane when out of bounds.
    fn shuffle_down(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        delta: Value,
        value_ty: TypeHandle,
    ) -> Value {
        let lane = self.lane_id(ctx, rw);
        let shifted = add(ctx, rw, lane, delta);
        let plane_dim = ctx.plane_dim() as i32;
        let end = insert_i32_const(ctx, rw, plane_dim);
        let has_source = icmp(ctx, rw, ICmpPredicateAttr::ULT, shifted, end);
        let src = select(ctx, rw, has_source, shifted, lane);
        self.shuffle(ctx, rw, value, src, value_ty)
    }

    /// Whether the predicate holds on all active lanes.
    fn all(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        let holds = self.ballot_mask(ctx, rw, input);
        let running = self.active_lanes(ctx, rw);
        icmp(ctx, rw, ICmpPredicateAttr::EQ, holds, running)
    }

    /// Whether the predicate holds on any active lane.
    fn any(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        let holds = self.ballot_mask(ctx, rw, input);
        let zero = mask_const(ctx, rw, 0);
        icmp(ctx, rw, ICmpPredicateAttr::NE, holds, zero)
    }

    /// Whether this is the elected lane.
    fn elect(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let running = self.active_lanes(ctx, rw);
        let mask_ty = mask_ty(ctx);
        let poison_if_zero = insert_bool_const(ctx, rw, false);
        let name = format!("{CTTZ}.{}", llvm_mangled_ty(ctx, mask_ty));
        let lowest = call_intrinsic(ctx, rw, &name, mask_ty, vec![running, poison_if_zero]);

        let lane = self.lane_id(ctx, rw);
        let lane = extend_to_mask(ctx, rw, lane);
        icmp(ctx, rw, ICmpPredicateAttr::EQ, lowest, lane)
    }
}

fn lowering(ctx: &Context) -> Option<Box<dyn PlaneLowering>> {
    match ctx.target() {
        #[cfg(feature = "amdgpu")]
        LlvmTarget::AmdGpu => Some(Box::new(crate::amdgpu::plane::AmdGpuPlane)),
        #[cfg(feature = "nvptx")]
        LlvmTarget::Nvptx => Some(Box::new(crate::nvptx::plane::NvptxPlane)),
        LlvmTarget::Cpu => None,
    }
}

macro_rules! lower_plane_op {
    ($cube_op:ty, |$this:ident, $lowering:ident, $ctx:ident, $rw:ident, $info:ident| $body:block) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                operands_info: &OperandsInfo,
            ) -> Result<()> {
                let Some($lowering) = lowering(ctx) else {
                    return input_err!(self.loc(ctx), PlaneOpsUnsupported);
                };
                let $this = self;
                let $ctx = ctx;
                let $rw = rewriter;
                #[allow(unused_variables)]
                let $info = operands_info;
                $body
            }
        }
    };
}

macro_rules! lower_shuffle {
    ($cube_op:ty, $method:ident, $operand:ident) => {
        lower_plane_op!($cube_op, |op, lowering, ctx, rw, info| {
            let old_op = op.get_operation();
            let input = op.input(ctx);
            let input_ty = operand_ty(ctx, info, input);
            let operand = op.$operand(ctx);

            let routed = lowering.$method(ctx, rw, input, operand, input_ty);
            rw.replace_operation_with_values(ctx, old_op, vec![routed]);
            Ok(())
        });
    };
}

lower_plane_op!(ShuffleOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let input_ty = operand_ty(ctx, info, input);
    let lane = op.lane(ctx);

    let routed = lowering.shuffle(ctx, rw, input, lane, input_ty);
    rw.replace_operation_with_values(ctx, old_op, vec![routed]);
    Ok(())
});

lower_shuffle!(ShuffleXorOp, shuffle_xor, mask);
lower_shuffle!(ShuffleUpOp, shuffle_up, delta);
lower_shuffle!(ShuffleDownOp, shuffle_down, delta);

lower_plane_op!(BroadcastOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let input_ty = operand_ty(ctx, info, input);
    let lane = op.lane(ctx).0 as i32;

    let src_lane = insert_i32_const(ctx, rw, lane);
    let routed = lowering.shuffle(ctx, rw, input, src_lane, input_ty);
    rw.replace_operation_with_values(ctx, old_op, vec![routed]);
    Ok(())
});

lower_plane_op!(AllOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let all = lowering.all(ctx, rw, input);
    rw.replace_operation_with_values(ctx, old_op, vec![all]);
    Ok(())
});

lower_plane_op!(AnyOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let any = lowering.any(ctx, rw, input);
    rw.replace_operation_with_values(ctx, old_op, vec![any]);
    Ok(())
});

lower_plane_op!(ElectOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let elected = lowering.elect(ctx, rw);
    rw.replace_operation_with_values(ctx, old_op, vec![elected]);
    Ok(())
});

lower_plane_op!(BallotOp, |op, lowering, ctx, rw, info| {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let mask = lowering.ballot_mask(ctx, rw, input);

    let i32_ty = i32_ty(ctx);
    let words = ctx.plane_dim() / 32;
    let vec_ty = LlvmVectorType::get(ctx, i32_ty, 4, VectorTypeKind::Fixed).into();

    let zero = insert_i32_const(ctx, rw, 0);
    let mut acc = insert_splat(ctx, rw, vec_ty, zero, 4);
    for word in 0..words {
        let shifted = if word == 0 {
            mask
        } else {
            let shift = mask_const(ctx, rw, (word * 32) as i128);
            lshr(ctx, rw, mask, shift)
        };
        let low = if ctx.plane_dim() == 32 {
            shifted
        } else {
            let trunc = llvm::TruncOp::new(ctx, shifted, i32_ty);
            insert(ctx, rw, &trunc)
        };
        let index = insert_i32_const(ctx, rw, word as i32);
        let op = llvm::InsertElementOp::new(ctx, acc, low, index);
        acc = insert(ctx, rw, &op);
    }

    rw.replace_operation_with_values(ctx, old_op, vec![acc]);
    Ok(())
});

/// Routes `value` through `route` one 32-bit word at a time and reassembles it: the plane
/// intrinsics of both GPU targets move a single 32-bit register per call.
pub fn route_words(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    value_ty: TypeHandle,
    mut route: impl FnMut(&mut Context, &mut DialectConversionRewriter, Value) -> Value,
) -> Value {
    let i32_ty = i32_ty(ctx);
    let llvm_ty = cube_type_to_llvm(ctx, value_ty);
    let bits = value_ty.size_bits(ctx) as u32;
    let words = bits.div_ceil(32);

    if words == 1 {
        let as_i32 = widen_to_i32(ctx, rw, value, bits);
        let routed = route(ctx, rw, as_i32);
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
        let routed = route(ctx, rw, word_value);
        let op = llvm::InsertElementOp::new(ctx, acc, routed, index);
        acc = insert(ctx, rw, &op);
    }
    bitcast(ctx, rw, acc, llvm_ty)
}

pub fn mask_ty(ctx: &mut Context) -> TypeHandle {
    let width = ctx.plane_dim();
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

pub fn mask_const(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: i128) -> Value {
    let width = ctx.plane_dim();
    insert_int_const(ctx, rw, width, value)
}

pub fn extend_to_mask(ctx: &mut Context, rw: &mut DialectConversionRewriter, lane: Value) -> Value {
    if ctx.plane_dim() == 32 {
        return lane;
    }
    let ty = mask_ty(ctx);
    let op = llvm::ZExtOp::new_with_nneg(ctx, lane, ty, false);
    insert(ctx, rw, &op)
}

pub fn call_intrinsic(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    ret_ty: TypeHandle,
    args: Vec<Value>,
) -> Value {
    let op = call_op(ctx, name, ret_ty, args);
    insert(ctx, rw, &op)
}

pub fn operand_ty(ctx: &Context, info: &OperandsInfo, value: Value) -> TypeHandle {
    info.lookup_most_recent_type(value)
        .unwrap_or_else(|| value.get_type(ctx))
}

pub fn bitcast(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    to: TypeHandle,
) -> Value {
    if value.get_type(ctx) == to {
        return value;
    }
    let op = llvm::BitcastOp::new(ctx, value, to);
    insert(ctx, rw, &op)
}

pub fn widen_to_i32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    bits: u32,
) -> Value {
    let i32_ty = i32_ty(ctx);
    if bits == 32 {
        return bitcast(ctx, rw, value, i32_ty);
    }
    let narrow_ty = IntegerType::get(ctx, bits, Signedness::Signless).into();
    let as_int = bitcast(ctx, rw, value, narrow_ty);
    let zext = llvm::ZExtOp::new_with_nneg(ctx, as_int, i32_ty, false);
    insert(ctx, rw, &zext)
}

pub fn narrow_from_i32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    bits: u32,
    result_ty: TypeHandle,
) -> Value {
    if bits == 32 {
        return bitcast(ctx, rw, value, result_ty);
    }
    let narrow_ty = IntegerType::get(ctx, bits, Signedness::Signless).into();
    let trunc = llvm::TruncOp::new(ctx, value, narrow_ty);
    let narrowed = insert(ctx, rw, &trunc);
    bitcast(ctx, rw, narrowed, result_ty)
}

pub fn icmp(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    predicate: ICmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = llvm::ICmpOp::new(ctx, predicate, lhs, rhs);
    insert(ctx, rw, &op)
}

pub fn select(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    condition: Value,
    on_true: Value,
    on_false: Value,
) -> Value {
    let op = llvm::SelectOp::new(ctx, condition, on_true, on_false);
    insert(ctx, rw, &op)
}

macro_rules! lane_arith {
    ($(#[$doc:meta])* $name:ident, bitwise, $op:path) => {
        $(#[$doc])*
        pub fn $name(
            ctx: &mut Context,
            rw: &mut DialectConversionRewriter,
            lhs: Value,
            rhs: Value,
        ) -> Value {
            let op = <$op>::new(ctx, lhs, rhs);
            insert(ctx, rw, &op)
        }
    };
    ($(#[$doc:meta])* $name:ident, arith, $op:path) => {
        $(#[$doc])*
        pub fn $name(
            ctx: &mut Context,
            rw: &mut DialectConversionRewriter,
            lhs: Value,
            rhs: Value,
        ) -> Value {
            let op = <$op>::new_with_overflow_flag(
                ctx,
                lhs,
                rhs,
                IntegerOverflowFlagsAttr::default(),
            );
            insert(ctx, rw, &op)
        }
    };
}

lane_arith!(xor, bitwise, llvm::XorOp);
lane_arith!(lshr, bitwise, llvm::LShrOp);
lane_arith!(shl, arith, llvm::ShlOp);
lane_arith!(add, arith, llvm::AddOp);
lane_arith!(sub, arith, llvm::SubOp);
