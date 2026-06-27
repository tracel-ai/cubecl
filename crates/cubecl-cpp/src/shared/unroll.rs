use cubecl_core::ir::{
    dialect::vector::{VectorExtractOp, VectorInitOp},
    interfaces::{MaterializableOp, TypedExt},
    prelude::*,
    rewrite::MatchRewritePass,
    types::VectorType,
};

#[op_interface]
pub trait UnrollingOp: MaterializableOp + OneResultInterface {
    verify_op_succ!();
}

macro_rules! unrolling {
    ($ty: ty) => {
        #[op_interface_impl]
        impl crate::shared::unroll::UnrollingOp for $ty {}
    };
}
pub(crate) use unrolling;

pub type CppUnrollPass = MatchRewritePass<CppUnroll>;

// Different implementation because the semantics are a bit different than the full unroll pass.
// It's also much simpler.
#[derive(Default, Clone)]
pub struct CppUnroll;

impl MatchRewrite for CppUnroll {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn UnrollingOp>(ctx) && op.result(ctx).is_vector(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let unroll_op = op_cast::<dyn UnrollingOp>(&*dyn_op).unwrap();
        let opds = op.operands(ctx);
        let attributes = op.deref(ctx).attributes.clone();
        let res = unroll_op.get_result(ctx);

        let vec_ty = {
            let ty = res.get_type(ctx).deref(ctx);
            *ty.downcast_ref::<VectorType>().unwrap()
        };

        let extract = |ctx: &mut Context, opd: &Value, i: usize| {
            if opd.vector_size(ctx) == 1 {
                // Scalar arg for things like lane index in plane ops. SameOperandTypes should
                // validate other args for equality so we don't get implicit broadcasts
                *opd
            } else {
                let extract = VectorExtractOp::new(ctx, *opd, i);
                extract.get_operation().insert_before(ctx, op);
                extract.get_result(ctx)
            }
        };
        let run_one = |ctx: &mut Context, i: usize| {
            let opds = opds.iter().map(|opd| extract(ctx, opd, i)).collect();
            let attrs = attributes.clone();
            let new_op = unroll_op.materialize(ctx, vec![vec_ty.inner], opds, attrs);
            new_op.insert_before(ctx, op);
            new_op.result(ctx)
        };

        let new_values = (0..vec_ty.vectorization).map(|i| run_one(ctx, i)).collect();
        let new_vec = VectorInitOp::new(ctx, new_values);
        new_vec.get_operation().insert_before(ctx, op);
        rewriter.replace_operation(ctx, op, new_vec.get_operation());

        Ok(())
    }
}
