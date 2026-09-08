//! Building calls to LLVM intrinsics, as every target's lowering needs them.
//!
//! Nothing here inserts the ops it builds, because the callers insert differently: the
//! builtins passes append to a [`Scope`](cubecl_core::ir::Scope), while the dialect
//! conversions insert before the op they replace.
//!
//! Which intrinsic to call is the target's business — `llvm.amdgcn.*` on one,
//! `llvm.nvvm.*` on the other — but the call itself is not.

use pliron_llvm::ops::CallIntrinsicOp;

use crate::shared::to_llvm::prelude::*;

/// Signless `i32`: what the intrinsics of both GPU targets take and return, and the type every
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
