use cubecl_core::ir::{
    dialect::general::ReinterpretCastOp,
    interfaces::{TypedExt, ValueExt},
    prelude::*,
    rewrite::MatchRewritePass,
};

#[op_interface]
pub trait PackableOp: OneResultInterface {
    verify_op_succ!();
    fn should_pack(&self, ctx: &Context) -> bool;
}

macro_rules! packable {
    ($ty: ty) => {
        #[op_interface_impl]
        impl crate::cuda::packed_ops::PackableOp for $ty {
            fn should_pack(&self, ctx: &pliron::context::Context) -> bool {
                use crate::shared::ty::TypedExtCPP;
                self.get_result(ctx).can_pack(ctx)
            }
        }
    };
}
pub(crate) use packable;
use pliron::irbuild::inserter::Inserter;

use crate::shared::ty::TypedExtCPP;

pub type PackOpsPass = MatchRewritePass<PackOps>;

#[derive(Default, Clone)]
pub struct PackOps;

impl MatchRewrite for PackOps {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn PackableOp>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_pack(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let unroll_op = op_cast::<dyn PackableOp>(&*dyn_op).unwrap();
        let res = unroll_op.get_result(ctx);
        let res_ty = res.get_type(ctx);

        for r#use in op.operands_as_uses(ctx) {
            let value = r#use.get_def(ctx);
            // Skip scalar arg in plane ops
            if value.vector_size(ctx) > 1 {
                let reinterpret = ReinterpretCastOp::new(ctx, value.packed_type(ctx), value);
                value.replace_use_with(ctx, r#use, &reinterpret.get_result(ctx));
                reinterpret.get_operation().insert_before(ctx, op);
            }
        }

        rewriter.set_value_type(ctx, res, res_ty.packed_type(ctx));
        let reinterpret_res = ReinterpretCastOp::new(ctx, res_ty, res);
        rewriter.insert_op(ctx, &reinterpret_res);
        let new_res = reinterpret_res.get_result(ctx);
        res.replace_all_uses_except_with(ctx, reinterpret_res.input_as_use(ctx), &new_res);

        Ok(())
    }
}
