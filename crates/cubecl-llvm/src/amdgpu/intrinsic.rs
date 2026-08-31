//! Calling the `llvm.amdgcn` intrinsics, and the lane index the cross-lane lowerings
//! start from.
//!
//! Two stages reach for these: [`builtins`](super::builtins) substitutes
//! `cube.read_builtin` while the cube dialect is still live, and the lowerings in
//! [`plane`](super::plane) and [`matrix`](super::matrix) run during the dialect
//! conversion. The two insert differently -- a [`Scope`](cubecl_core::ir::Scope) appends
//! to the block it is building, a rewriter inserts before the op it is replacing -- so
//! what is shared here is the ops themselves, and each caller inserts them its own way.

use pliron_llvm::ops::CallIntrinsicOp;

use crate::shared::to_llvm::prelude::*;

/// Counts the set bits of the exec mask below this lane, which under a full mask is the
/// lane's own index in the wavefront. Split in two halves so the same pair serves both
/// wave widths: on wave32 the high half adds nothing.
const MBCNT_LO: &str = "llvm.amdgcn.mbcnt.lo";
const MBCNT_HI: &str = "llvm.amdgcn.mbcnt.hi";

/// Signless `i32`: what every `amdgcn` intrinsic takes and returns, and the type every
/// cube integer converges to in the LLVM dialect.
///
/// Tagging these with cube's `u32` (`Signedness::Unsigned`) instead would leave them
/// unsigned forever while the constants they get paired with are forced signless,
/// tripping `SameOperandsType` verification despite representing the same value.
pub fn i32_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

/// A call to the LLVM intrinsic `name` over `args`, returning `ret_ty`.
///
/// `llvm.call_intrinsic` carries the name and type as attributes; the function
/// declaration is added lazily during `to_llvm_ir`, as `shared::to_llvm::math` does.
pub fn call_op(
    ctx: &mut Context,
    name: &str,
    ret_ty: TypeHandle,
    args: Vec<Value>,
) -> CallIntrinsicOp {
    let arg_tys = args.iter().map(|a| a.get_type(ctx)).collect();
    let fn_ty = FuncType::get(ctx, ret_ty, arg_tys, false);
    CallIntrinsicOp::new(ctx, name.into(), fn_ty, args)
}

/// A signless `i32` constant, which is what the intrinsics here take.
pub fn i32_const_op(ctx: &mut Context, value: i32) -> llvm::ConstantOp {
    let attr = int_attr(ctx, I32_WIDTH, value as i128);
    llvm::ConstantOp::new(ctx, attr.into())
}

/// This lane's index within its wavefront: the operations that compute it, in the order
/// they must be inserted, and the value they produce.
///
/// A result is readable as soon as its operation is built, so the whole chain is built
/// here and the caller decides where it lands.
pub fn lane_id_ops(ctx: &mut Context) -> (Vec<Ptr<Operation>>, Value) {
    let ty = i32_ty(ctx);
    let all_lanes = i32_const_op(ctx, -1);
    let zero = i32_const_op(ctx, 0);

    let lo = call_op(
        ctx,
        MBCNT_LO,
        ty,
        vec![all_lanes.get_result(ctx), zero.get_result(ctx)],
    );
    let hi = call_op(
        ctx,
        MBCNT_HI,
        ty,
        vec![all_lanes.get_result(ctx), lo.get_result(ctx)],
    );

    let lane = hi.get_result(ctx);
    let ops = vec![
        all_lanes.get_operation(),
        zero.get_operation(),
        lo.get_operation(),
        hi.get_operation(),
    ];
    (ops, lane)
}
