//! Lowering of the plane operations to the wavefront's cross-lane instructions.
//!
//! A plane is a wavefront, and every shuffle here is `ds_bpermute`: each lane names the lane it
//! wants to read from and the hardware routes the value. It moves 32 bits at a time, so wider
//! and narrower values are taken apart into words, routed, and put back together.
//!
//! `ballot` is the exec mask of a predicate, which `all`, `any` and `elect` are then questions
//! about. The reductions and scans are not here at all: they are polyfills over these shuffles,
//! shared with the C++ backends in [`polyfills::plane`](cubecl_core::prelude::polyfills::plane).

use cubecl_core::ir::ContextExt;
use cubecl_core::ir::dialect::plane::{
    AllOp, AnyOp, BallotOp, BroadcastOp, ElectOp, ShuffleDownOp, ShuffleOp, ShuffleUpOp,
    ShuffleXorOp,
};

use crate::amdgpu::intrinsic::{call_op, i32_ty, lane_id_ops};
use crate::shared::to_llvm::prelude::*;

/// Routes a 32-bit word between the lanes of a wavefront. The address is a byte address, so a
/// lane index has to be scaled by four.
const DS_BPERMUTE: &str = "llvm.amdgcn.ds.bpermute";

/// Wavefront width of the device, which the shuffles need to know where a plane ends.
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

/// Emits a call to the intrinsic `name` over `args`.
fn call(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    name: &str,
    ret_ty: TypeHandle,
    args: Vec<Value>,
) -> Value {
    let op = call_op(ctx, name, ret_ty, args);
    insert(ctx, rewriter, &op)
}

/// This lane's index within its wavefront.
///
/// Recomputed rather than taken from the builtin, which has long since been substituted by the
/// time the ops reach here. The optimizer folds the duplicates back together.
pub(crate) fn lane_id(ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Value {
    let (ops, lane) = lane_id_ops(ctx);
    for op in ops {
        rewriter.insert_operation(ctx, op);
    }
    lane
}

/// Emits a bitwise `llvm` op over two `i32`.
macro_rules! bitwise {
    ($ctx:expr, $rewriter:expr, $op:path, $lhs:expr, $rhs:expr) => {{
        let op = <$op>::new($ctx, $lhs, $rhs);
        insert($ctx, $rewriter, &op)
    }};
}

/// Emits an arithmetic `llvm` op over two `i32`. These carry overflow flags, which are left at
/// their default: a lane index cannot overflow anything.
macro_rules! arith {
    ($ctx:expr, $rewriter:expr, $op:path, $lhs:expr, $rhs:expr) => {{
        let op =
            <$op>::new_with_overflow_flag($ctx, $lhs, $rhs, IntegerOverflowFlagsAttr::default());
        insert($ctx, $rewriter, &op)
    }};
}

/// Routes `value` from the lane `src_lane` of this wavefront.
///
/// `ds_bpermute` moves one word, so anything else is bitcast to a whole number of words, routed
/// a word at a time, and bitcast back.
fn shuffle(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    src_lane: Value,
    value_ty: TypeHandle,
) -> Value {
    let i32_ty = i32_ty(ctx);
    let two = insert_i32_const(ctx, rewriter, 2);
    let addr = arith!(ctx, rewriter, llvm::ShlOp, src_lane, two);

    let llvm_ty = cube_type_to_llvm(ctx, value_ty);
    let bits = value_ty.size_bits(ctx) as u32;
    let words = bits.div_ceil(32);

    // One word: route the bit pattern straight through.
    if words == 1 {
        let as_i32 = widen_to_i32(ctx, rewriter, value, bits);
        let routed = call(ctx, rewriter, DS_BPERMUTE, i32_ty, vec![addr, as_i32]);
        return narrow_from_i32(ctx, rewriter, routed, bits, llvm_ty);
    }

    // Several: through a vector of words, so each element is one routing.
    let words_ty = LlvmVectorType::get(ctx, i32_ty, words, VectorTypeKind::Fixed).into();
    let as_words = bitcast(ctx, rewriter, value, words_ty);

    let poison = llvm::PoisonOp::new(ctx, words_ty);
    let mut acc = insert(ctx, rewriter, &poison);
    for word in 0..words {
        let index = insert_i32_const(ctx, rewriter, word as i32);
        let extract = llvm::ExtractElementOp::new(ctx, as_words, index);
        let word_value = insert(ctx, rewriter, &extract);
        let one = call(ctx, rewriter, DS_BPERMUTE, i32_ty, vec![addr, word_value]);
        let op = llvm::InsertElementOp::new(ctx, acc, one, index);
        acc = insert(ctx, rewriter, &op);
    }
    bitcast(ctx, rewriter, acc, llvm_ty)
}

/// `value` as one `i32`, whatever its own type is.
fn widen_to_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    bits: u32,
) -> Value {
    let i32_ty = i32_ty(ctx);
    if bits == 32 {
        return bitcast(ctx, rewriter, value, i32_ty);
    }
    let narrow_ty = IntegerType::get(ctx, bits, Signedness::Signless).into();
    let as_int = bitcast(ctx, rewriter, value, narrow_ty);
    let zext = llvm::ZExtOp::new_with_nneg(ctx, as_int, i32_ty, false);
    insert(ctx, rewriter, &zext)
}

/// The inverse of [`widen_to_i32`].
fn narrow_from_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    bits: u32,
    result_ty: TypeHandle,
) -> Value {
    if bits == 32 {
        return bitcast(ctx, rewriter, value, result_ty);
    }
    let narrow_ty = IntegerType::get(ctx, bits, Signedness::Signless).into();
    let trunc = llvm::TruncOp::new(ctx, value, narrow_ty);
    let narrowed = insert(ctx, rewriter, &trunc);
    bitcast(ctx, rewriter, narrowed, result_ty)
}

/// A bitcast, skipped when it would be to the type the value already has.
fn bitcast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    to: TypeHandle,
) -> Value {
    if value.get_type(ctx) == to {
        return value;
    }
    let op = llvm::BitcastOp::new(ctx, value, to);
    insert(ctx, rewriter, &op)
}

/// The type an operand carries at this point in the conversion.
fn operand_ty(ctx: &Context, info: &OperandsInfo, value: Value) -> TypeHandle {
    info.lookup_most_recent_type(value)
        .unwrap_or_else(|| value.get_type(ctx))
}

/// Lowers a shuffle whose source lane is `$src`, given `lane` and the op's own operand.
macro_rules! lower_shuffle {
    ($cube_op:ty, $operand:ident, |$ctx:ident, $rewriter:ident, $lane:ident, $operand_val:ident| $src:block) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                operands_info: &OperandsInfo,
            ) -> Result<()> {
                let old_op = self.get_operation();
                let input = self.input(ctx);
                let input_ty = operand_ty(ctx, operands_info, input);
                #[allow(unused_variables)]
                let $operand_val = self.$operand(ctx);

                let src_lane = {
                    let $ctx = &mut *ctx;
                    let $rewriter = &mut *rewriter;
                    let $lane = lane_id($ctx, $rewriter);
                    $src
                };

                let routed = shuffle(ctx, rewriter, input, src_lane, input_ty);
                rewriter.replace_operation_with_values(ctx, old_op, vec![routed]);
                Ok(())
            }
        }
    };
}

// An absolute lane, so nothing to compute: the value is read from where it is asked for.
lower_shuffle!(ShuffleOp, lane, |ctx, rewriter, lane, operand| {
    let _ = lane;
    operand
});

// A butterfly: the partner lane differs from this one in the bits of the mask.
lower_shuffle!(ShuffleXorOp, mask, |ctx, rewriter, lane, operand| {
    bitwise!(ctx, rewriter, llvm::XorOp, lane, operand)
});

// Reading from a lower lane, and from itself where there is no lower lane to read.
lower_shuffle!(ShuffleUpOp, delta, |ctx, rewriter, lane, operand| {
    let shifted = arith!(ctx, rewriter, llvm::SubOp, lane, operand);
    let has_source = icmp(ctx, rewriter, ICmpPredicateAttr::UGE, lane, operand);
    select(ctx, rewriter, has_source, shifted, lane)
});

// The same upward, bounded by the end of the plane.
lower_shuffle!(ShuffleDownOp, delta, |ctx, rewriter, lane, operand| {
    let shifted = arith!(ctx, rewriter, llvm::AddOp, lane, operand);
    let plane_dim = ctx.plane_dim() as i32;
    let end = insert_i32_const(ctx, rewriter, plane_dim);
    let has_source = icmp(ctx, rewriter, ICmpPredicateAttr::ULT, shifted, end);
    select(ctx, rewriter, has_source, shifted, lane)
});

#[op_interface_impl]
impl ToLLVMDialect for BroadcastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let input = self.input(ctx);
        let input_ty = operand_ty(ctx, operands_info, input);
        // The lane is an attribute here rather than a value, so it is a constant to build.
        let lane = self.lane(ctx).0 as i32;

        let src_lane = insert_i32_const(ctx, rewriter, lane);
        let routed = shuffle(ctx, rewriter, input, src_lane, input_ty);
        rewriter.replace_operation_with_values(ctx, old_op, vec![routed]);
        Ok(())
    }
}

/// An integer comparison between two `i32`.
fn icmp(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    predicate: ICmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = llvm::ICmpOp::new(ctx, predicate, lhs, rhs);
    insert(ctx, rewriter, &op)
}

/// `condition ? on_true : on_false`.
fn select(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    condition: Value,
    on_true: Value,
    on_false: Value,
) -> Value {
    let op = llvm::SelectOp::new(ctx, condition, on_true, on_false);
    insert(ctx, rewriter, &op)
}

/// The exec mask of a predicate: one bit per lane of the wavefront, set where it holds.
const BALLOT: &str = "llvm.amdgcn.ballot";

/// Counts the trailing zeros of an integer, i.e. finds the lowest set bit.
const CTTZ: &str = "llvm.cttz";

/// An integer as wide as the wavefront, which is what a ballot returns.
fn mask_ty(ctx: &mut Context) -> TypeHandle {
    let width = ctx.plane_dim();
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

/// The mask of the lanes where `predicate` holds.
fn ballot(ctx: &mut Context, rewriter: &mut DialectConversionRewriter, predicate: Value) -> Value {
    let ty = mask_ty(ctx);
    // The intrinsic is overloaded on its return type, so the mangled name carries the width.
    let name = format!("{BALLOT}.{}", llvm_mangled_ty(ctx, ty));
    call(ctx, rewriter, &name, ty, vec![predicate])
}

/// The mask of the lanes that are running at all.
fn active_lanes(ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Value {
    let all = insert_bool_const(ctx, rewriter, true);
    ballot(ctx, rewriter, all)
}

/// A mask constant of the wavefront's width.
fn mask_const(ctx: &mut Context, rewriter: &mut DialectConversionRewriter, value: i128) -> Value {
    let width = ctx.plane_dim();
    insert_int_const(ctx, rewriter, width, value)
}

#[op_interface_impl]
impl ToLLVMDialect for AllOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let input = self.input(ctx);

        // True everywhere it is running, which is what makes this `all` and not `any`.
        let holds = ballot(ctx, rewriter, input);
        let running = active_lanes(ctx, rewriter);
        let all = icmp(ctx, rewriter, ICmpPredicateAttr::EQ, holds, running);

        rewriter.replace_operation_with_values(ctx, old_op, vec![all]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for AnyOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let input = self.input(ctx);

        let holds = ballot(ctx, rewriter, input);
        let zero = mask_const(ctx, rewriter, 0);
        let any = icmp(ctx, rewriter, ICmpPredicateAttr::NE, holds, zero);

        rewriter.replace_operation_with_values(ctx, old_op, vec![any]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for ElectOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();

        // The lowest lane still running, and only it, answers yes.
        let running = active_lanes(ctx, rewriter);
        let mask_ty = mask_ty(ctx);
        let poison_if_zero = insert_bool_const(ctx, rewriter, false);
        let name = format!("{CTTZ}.{}", llvm_mangled_ty(ctx, mask_ty));
        let lowest = call(ctx, rewriter, &name, mask_ty, vec![running, poison_if_zero]);

        let lane = lane_id(ctx, rewriter);
        let lane = extend_to_mask(ctx, rewriter, lane);
        let elected = icmp(ctx, rewriter, ICmpPredicateAttr::EQ, lowest, lane);

        rewriter.replace_operation_with_values(ctx, old_op, vec![elected]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for BallotOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let input = self.input(ctx);
        let mask = ballot(ctx, rewriter, input);

        // The result is four words whatever the wavefront's width, so the mask goes in the
        // low ones and the rest stay clear.
        let i32_ty = i32_ty(ctx);
        let words = ctx.plane_dim() / 32;
        let vec_ty = LlvmVectorType::get(ctx, i32_ty, 4, VectorTypeKind::Fixed).into();

        let zero = insert_i32_const(ctx, rewriter, 0);
        let mut acc = insert_splat(ctx, rewriter, vec_ty, zero, 4);
        for word in 0..words {
            let shifted = if word == 0 {
                mask
            } else {
                let shift = mask_const(ctx, rewriter, (word * 32) as i128);
                bitwise!(ctx, rewriter, llvm::LShrOp, mask, shift)
            };
            let low = if ctx.plane_dim() == 32 {
                shifted
            } else {
                let trunc = llvm::TruncOp::new(ctx, shifted, i32_ty);
                insert(ctx, rewriter, &trunc)
            };
            let index = insert_i32_const(ctx, rewriter, word as i32);
            let op = llvm::InsertElementOp::new(ctx, acc, low, index);
            acc = insert(ctx, rewriter, &op);
        }

        rewriter.replace_operation_with_values(ctx, old_op, vec![acc]);
        Ok(())
    }
}

/// Widens a lane index to the width a ballot mask is compared at.
fn extend_to_mask(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    lane: Value,
) -> Value {
    if ctx.plane_dim() == 32 {
        return lane;
    }
    let ty = mask_ty(ctx);
    let op = llvm::ZExtOp::new_with_nneg(ctx, lane, ty, false);
    insert(ctx, rewriter, &op)
}
