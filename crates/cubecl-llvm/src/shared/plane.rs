//! Lowering of the plane operations, as the two GPU targets share them.
//!
//! A plane is a wavefront on AMD and a warp on NVIDIA, and the two hardware families expose
//! very nearly the same set of primitives over one: route a value from a named lane, and take
//! the mask of lanes where a predicate holds. Everything the cube dialect asks for is built
//! from those, so this module holds the ops once and [`PlaneLowering`] holds what differs.
//!
//! The derived operations come with default bodies, which is what a new target gets for free:
//! `all` is a ballot compared against the running lanes, the directional shuffles are lane
//! arithmetic in front of an absolute one. A target with an instruction of its own for any of
//! them overrides that method rather than reimplementing the op.
//!
//! The reductions and scans are not here at all: they are polyfills over these shuffles, in
//! [`plane_reduce`](super::plane_reduce).

use cubecl_core::ir::ContextExt;
use cubecl_core::ir::dialect::plane::{
    AllOp, AnyOp, BallotOp, BroadcastOp, ElectOp, ShuffleDownOp, ShuffleOp, ShuffleUpOp,
    ShuffleXorOp,
};
use pliron::input_err;
use thiserror::Error;

use crate::shared::intrinsic::{call_op, i32_ty};
use crate::shared::to_llvm::prelude::*;
use crate::target::{CtxTarget, LlvmTarget};

/// Counts the trailing zeros of an integer, i.e. finds the lowest set bit.
pub(crate) const CTTZ: &str = "llvm.cttz";

/// Width of the plane, which the shuffles need to know where a plane ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneDim(pub u32);

impl CtxPlaneDim for Context {}

/// The plane width on the context.
pub trait CtxPlaneDim: ContextExt {
    fn plane_dim(&self) -> u32 {
        self.aux_ty::<PlaneDim>().0
    }
    fn set_plane_dim(&mut self, plane_dim: u32) {
        self.set_aux_ty(PlaneDim(plane_dim));
    }
}

/// A plane operation on a target that has no plane wider than one unit.
#[derive(Debug, Error)]
#[error(
    "plane operations need a plane of more than one unit, which the CPU target does not have; \
     a runtime that lowers them here has advertised `Plane::Ops` it cannot honour"
)]
pub struct PlaneOpsUnsupported;

/// What one target contributes to the plane operations.
///
/// The three required methods are the hardware primitives; the rest are derived from them and
/// overridden only where a target has something better.
pub trait PlaneLowering {
    /// This lane's index within its plane.
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value;

    /// `value` as held by the lane `src_lane` of this plane, whatever its type.
    fn shuffle(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        value: Value,
        src_lane: Value,
        value_ty: TypeHandle,
    ) -> Value;

    /// The mask of the lanes where `predicate` holds, as an integer of the plane's width.
    fn ballot_mask(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        predicate: Value,
    ) -> Value;

    /// The mask of the lanes that are running at all.
    fn active_lanes(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        let all = insert_bool_const(ctx, rw, true);
        self.ballot_mask(ctx, rw, all)
    }

    /// A butterfly: the partner lane differs from this one in the bits of `mask`.
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

    /// Reading from a lower lane, and from itself where there is no lower lane to read.
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

    /// The same upward, bounded by the end of the plane.
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

    /// Whether `predicate` holds on every lane that is running.
    fn all(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        // True everywhere it is running, which is what makes this `all` and not `any`.
        let holds = self.ballot_mask(ctx, rw, input);
        let running = self.active_lanes(ctx, rw);
        icmp(ctx, rw, ICmpPredicateAttr::EQ, holds, running)
    }

    /// Whether `predicate` holds on any lane that is running.
    fn any(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter, input: Value) -> Value {
        let holds = self.ballot_mask(ctx, rw, input);
        let zero = mask_const(ctx, rw, 0);
        icmp(ctx, rw, ICmpPredicateAttr::NE, holds, zero)
    }

    /// Whether this is the one lane the plane elects.
    fn elect(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        // The lowest lane still running, and only it, answers yes.
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

/// The plane lowering for the target the context is compiling for.
fn lowering(ctx: &Context) -> Option<Box<dyn PlaneLowering>> {
    match ctx.target() {
        LlvmTarget::AmdGpu => Some(Box::new(crate::amdgpu::plane::AmdGpuPlane)),
        LlvmTarget::Nvptx => Some(Box::new(crate::nvptx::plane::NvptxPlane)),
        LlvmTarget::Cpu => None,
    }
}

/// Lowers a plane op through the target's [`PlaneLowering`], reporting the CPU's absence of a
/// plane as an error on the op rather than as a panic.
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

/// The four shuffles, which differ only in the lane each one reads from.
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

// An absolute lane, so nothing to derive: the value is read from where it is asked for.
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

// The lane is an attribute here rather than a value, so it is a constant to build.
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

    // The result is four words whatever the plane's width, so the mask goes in the low ones
    // and the rest stay clear.
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

/// An integer as wide as the plane, which is what a ballot returns.
pub fn mask_ty(ctx: &mut Context) -> TypeHandle {
    let width = ctx.plane_dim();
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

/// A mask constant of the plane's width.
pub fn mask_const(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: i128) -> Value {
    let width = ctx.plane_dim();
    insert_int_const(ctx, rw, width, value)
}

/// Widens a lane index to the width a ballot mask is compared at.
pub fn extend_to_mask(ctx: &mut Context, rw: &mut DialectConversionRewriter, lane: Value) -> Value {
    if ctx.plane_dim() == 32 {
        return lane;
    }
    let ty = mask_ty(ctx);
    let op = llvm::ZExtOp::new_with_nneg(ctx, lane, ty, false);
    insert(ctx, rw, &op)
}

/// Emits a call to the LLVM intrinsic `name` over `args`, inserted before the op being
/// replaced.
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

/// The type an operand carries at this point in the conversion.
pub fn operand_ty(ctx: &Context, info: &OperandsInfo, value: Value) -> TypeHandle {
    info.lookup_most_recent_type(value)
        .unwrap_or_else(|| value.get_type(ctx))
}

/// A bitcast, skipped when it would be to the type the value already has.
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

/// `value` as one `i32`, whatever its own type is.
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

/// The inverse of [`widen_to_i32`].
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

/// An integer comparison between two values of the same width.
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

/// `condition ? on_true : on_false`.
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

/// The bitwise and arithmetic ops the lane arithmetic above is built from. The arithmetic ones
/// carry overflow flags, left at their default: a lane index cannot overflow anything.
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

lane_arith!(
    /// Bitwise exclusive or.
    xor, bitwise, llvm::XorOp
);
lane_arith!(
    /// Logical right shift.
    lshr, bitwise, llvm::LShrOp
);
lane_arith!(
    /// Left shift.
    shl, arith, llvm::ShlOp
);
lane_arith!(
    /// Addition.
    add, arith, llvm::AddOp
);
lane_arith!(
    /// Subtraction.
    sub, arith, llvm::SubOp
);
