//! Shared LLVM intrinsic helpers.

use crate::prelude::*;
use pliron_llvm::ops::CallIntrinsicOp;

/// LLVM intrinsics use signless integers.
pub fn i32_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

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

pub fn i32_const_op(ctx: &mut Context, value: i32) -> llvm::ConstantOp {
    let attr = int_attr(ctx, I32_WIDTH, value as i128);
    llvm::ConstantOp::new(ctx, attr.into())
}
