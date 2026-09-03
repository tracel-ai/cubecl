use cubecl_ir::{
    NamedRewrite,
    dialect::math::{FAddOp, FMulOp, FNegOp, FSubOp, FmaOp},
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::MatchRewritePass,
};

/// Combine a product and the add or subtract that consumes it into a single FMA operation.
pub type InstCombinePass = MatchRewritePass<InstCombine>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct InstCombine;

/// Which operand a subtraction leaves needing its sign flipped.
#[derive(Clone, Copy)]
enum Negate {
    /// `a * b + c`.
    Nothing,
    /// `a * b - c`.
    Addend,
    /// `c - a * b`, where flipping one factor flips the product.
    Factor,
}

/// The product `value` defines, if this is its only use.
///
/// Fusing one that feeds anything else would leave it computed twice.
fn lone_product(ctx: &Context, value: Value) -> Option<FMulOp> {
    let mul = value.defining_op()?.as_op::<FMulOp>(ctx)?;
    (mul.get_result(ctx).num_uses(ctx) == 1).then_some(mul)
}

/// The product, what is added to it, and what needs negating, for an `op` of the form
/// `a * b + c`, `a * b - c` or `c - a * b`, in either operand order.
fn as_fma(ctx: &Context, op: Ptr<Operation>) -> Option<(FMulOp, Value, Negate)> {
    if let Some(add) = op.as_op::<FAddOp>(ctx) {
        let (lhs, rhs) = (add.lhs(ctx), add.rhs(ctx));
        return lone_product(ctx, lhs)
            .map(|mul| (mul, rhs, Negate::Nothing))
            .or_else(|| lone_product(ctx, rhs).map(|mul| (mul, lhs, Negate::Nothing)));
    }

    let sub = op.as_op::<FSubOp>(ctx)?;
    let (lhs, rhs) = (sub.lhs(ctx), sub.rhs(ctx));
    lone_product(ctx, lhs)
        .map(|mul| (mul, rhs, Negate::Addend))
        .or_else(|| lone_product(ctx, rhs).map(|mul| (mul, lhs, Negate::Factor)))
}

impl MatchRewrite for InstCombine {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        as_fma(ctx, op).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some((mul, addend, negate)) = as_fma(ctx, op) else {
            return Ok(());
        };
        let (mut a, b, mut c) = (mul.lhs(ctx), mul.rhs(ctx), addend);

        let flip = match negate {
            Negate::Nothing => None,
            Negate::Addend => Some(c),
            Negate::Factor => Some(a),
        };
        if let Some(value) = flip {
            let neg = FNegOp::new(ctx, value);
            rewriter.insert_op(ctx, &neg);
            match negate {
                Negate::Addend => c = neg.get_result(ctx),
                _ => a = neg.get_result(ctx),
            }
        }

        let fma = FmaOp::new(ctx, a, b, c);
        rewriter.insert_op(ctx, &fma);
        rewriter.replace_operation(ctx, op, fma.get_operation());
        rewriter.erase_operation(ctx, mul.get_operation());
        Ok(())
    }
}
