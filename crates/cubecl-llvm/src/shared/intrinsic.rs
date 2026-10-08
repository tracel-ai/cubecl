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
    llvm::ConstantOp::new(ctx, Box::new(attr))
}

pub fn int_ty(ctx: &mut Context, width: u32) -> TypeHandle {
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

pub fn i64_ty(ctx: &mut Context) -> TypeHandle {
    int_ty(ctx, 64)
}

/// `value`, an integer, truncated or extended to `width` bits, sign extended when `signed`.
/// `None` when `value` is not an integer.
pub fn resize_int(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    width: u32,
    signed: bool,
) -> Option<Value> {
    let from = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int| int.width())?;
    let ty = int_ty(ctx, width);
    let op: Ptr<Operation> = match from.cmp(&width) {
        core::cmp::Ordering::Equal => return Some(value),
        core::cmp::Ordering::Greater => llvm::TruncOp::new(ctx, value, ty).get_operation(),
        core::cmp::Ordering::Less if signed => llvm::SExtOp::new(ctx, value, ty).get_operation(),
        core::cmp::Ordering::Less => {
            llvm::ZExtOp::new_with_nneg(ctx, value, ty, false).get_operation()
        }
    };
    rw.insert_operation(ctx, op);
    Some(op.deref(ctx).get_result(0))
}
