//! Convert unsupported types ahead of time. Also convert auto-promoted types manually so we can
//! properly preserve the semantics of the actual code in IR.

use cubecl_core::ir::{
    NamedRewrite,
    dialect::{OperationPtrExt, general::CastOp},
    interfaces::ValueExt,
    prelude::{Context, MatchRewriter, Operation, Ptr, Result},
    rewrite::MatchRewritePass,
    types::scalar::Float32Type,
    verify_op_succ,
};
use pliron::{
    builtin::{
        op_interfaces::OneResultInterface,
        types::{IntegerType, Signedness},
    },
    derive::op_interface,
    irbuild::match_rewrite::MatchRewrite,
    op::Op,
    r#type::{TypeHandle, Typed},
    value::Value,
};

#[op_interface]
pub trait HalfPromotedOp: OneResultInterface {
    verify_op_succ!();
}

#[op_interface]
pub trait IntPromotedOp: OneResultInterface {
    verify_op_succ!();
}

macro_rules! no_half {
    ($ty: ty) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::convert::HalfPromotedOp for $ty {}
    };
}

pub(crate) use no_half;

/// An op whose MSL builtin has no `bfloat` overload. `bfloat` converts to `float` implicitly but
/// back only explicitly, so the call resolves to the `float` overload and its result cannot be
/// stored; on Metal a bf16 op of this kind runs in f32.
#[op_interface]
pub trait NoMslBfloatOp: OneResultInterface {
    verify_op_succ!();
}

macro_rules! no_msl_bfloat {
    ($ty: ty) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::convert::NoMslBfloatOp for $ty {}
    };
}
pub(crate) use no_msl_bfloat;

macro_rules! promotes_int {
    ($ty: ty) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::convert::IntPromotedOp for $ty {}
    };
}
pub(crate) use promotes_int;

use crate::{
    shared::ty::TypedExtCPP,
    target::{CtxTarget, Target},
};

pub type PromoteUnsupportedTypesPass = MatchRewritePass<PromoteUnsupportedTypes>;

#[derive(Default, NamedRewrite)]
pub struct PromoteUnsupportedTypes;

impl MatchRewrite for PromoteUnsupportedTypes {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        let promote_half = op.impls::<dyn HalfPromotedOp>(ctx) && op.result(ctx).is_half(ctx);
        let promote_int = op.impls::<dyn IntPromotedOp>(ctx) && op.result(ctx).is_small_int(ctx);
        promote_half || promote_int || promotes_msl_bfloat(ctx, op)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        _rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        if op.impls::<dyn HalfPromotedOp>(ctx) && op.result(ctx).is_half(ctx) {
            let f32 = Float32Type::get(ctx).to_handle();
            promote(ctx, op, |ctx, value| value.is_half(ctx), f32);
        }
        if promotes_msl_bfloat(ctx, op) {
            let f32 = Float32Type::get(ctx).to_handle();
            promote(ctx, op, |ctx, value| value.is_bf16(ctx), f32);
        }
        if op.impls::<dyn IntPromotedOp>(ctx) && op.result(ctx).is_small_signed_int(ctx) {
            let i32 = IntegerType::get(ctx, 32, Signedness::Signed).to_handle();
            promote(ctx, op, |ctx, value| value.is_small_int(ctx), i32);
        }
        if op.impls::<dyn IntPromotedOp>(ctx) && op.result(ctx).is_small_unsigned_int(ctx) {
            let u32 = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
            promote(ctx, op, |ctx, value| value.is_small_int(ctx), u32);
        }

        Ok(())
    }
}

fn promotes_msl_bfloat(ctx: &Context, op: Ptr<Operation>) -> bool {
    ctx.target() == Target::Metal
        && op.impls::<dyn NoMslBfloatOp>(ctx)
        && op.result(ctx).is_bf16(ctx)
}

fn promote(
    ctx: &mut Context,
    op: Ptr<Operation>,
    pred: impl Fn(&Context, Value) -> bool,
    promoted_type: TypeHandle,
) {
    let res = op.result(ctx);
    let res_ty = res.get_type(ctx);

    for r#use in op.operands_as_uses(ctx) {
        let value = r#use.get_def(ctx);
        if pred(ctx, value) {
            let cast = CastOp::new(ctx, promoted_type, value);
            cast.get_operation().insert_before(ctx, op);
            value.replace_use_with(ctx, r#use, &cast.get_result(ctx));
        }
    }

    res.set_type(ctx, promoted_type);
    let cast_res = CastOp::new(ctx, res_ty, res);
    let new_res = cast_res.get_result(ctx);
    cast_res.get_operation().insert_after(ctx, op);
    res.replace_all_uses_except_with(ctx, cast_res.input_as_use(ctx), &new_res);
}
