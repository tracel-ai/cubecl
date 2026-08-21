use cubecl_ir::{
    dialect::math::{FAddOp, FMulOp, FmaOp},
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::{MatchRewritePass},
    NamedRewrite,
};

/// Combine `a * b + c` or `c + a * b` into a single FMA operation.
pub type InstCombinePass = MatchRewritePass<InstCombine>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct InstCombine;

/// Returns the `(a, b, c)` of an fma if `add` is `a * b + c` in either operand order.
fn as_fma(ctx: &Context, add: FAddOp) -> Option<(FMulOp, Value, Value, Value)> {
    let fuse = |mul: Value, addend: Value| {
        let mul = mul.defining_op()?.as_op::<FMulOp>(ctx)?;
        // Only fuse when the product is dead afterwards, otherwise we'd compute it twice.
        (mul.get_result(ctx).num_uses(ctx) == 1)
            .then(|| (mul, mul.lhs(ctx), mul.rhs(ctx), addend))
    };
    let (lhs, rhs) = (add.lhs(ctx), add.rhs(ctx));
    fuse(lhs, rhs).or_else(|| fuse(rhs, lhs))
}

impl MatchRewrite for InstCombine {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.as_op::<FAddOp>(ctx)
            .is_some_and(|add| as_fma(ctx, add).is_some())
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let add = op.as_op::<FAddOp>(ctx).expect("matched an FAddOp");
        let Some((mul, a, b, c)) = as_fma(ctx, add) else {
            return Ok(());
        };
        let fma = FmaOp::new(ctx, a, b, c);
        rewriter.insert_op(ctx, &fma);
        rewriter.replace_operation(ctx, op, fma.get_operation());
        rewriter.erase_operation(ctx, mul.get_operation());
        Ok(())
    }
}
