//! Re-index a divisibility-guarded range loop by the quotient of its guard.
//!
//! ```text
//! for i in lo..hi { if (base - i * s) % d == 0 { body } }
//! ```
//!
//! runs `hi - lo` times but only reaches `body` on the `i` where `d` divides
//! `base - i * s`. Dilated conv-transpose is the motivating shape: the window
//! it walks is `kernel * dilation` wide and every `dilation`-th tap survives,
//! so the trip count grows with `dilation` while the arithmetic does not.
//!
//! The surviving taps are in bijection with the quotient
//! `q = (base - i * s) / d`, whose inverse is `i = (base - q * d) / s` wherever
//! `s` divides `base - q * d`. Driving the loop by `q` therefore reaches each
//! tap exactly once. The body is left alone; the rewrite only recovers `i` and
//! adds the checks that make the recovery sound, which is also what lets the
//! new bounds be a loose superset instead of an exact range.
//!
//! A zero `s` maps every `i` to one `q`, a zero `d` divides nothing, and an
//! `i * s` past `base` wraps the numerator onto a quotient the bounds do not
//! cover. Rather than keep a second copy of the loop for those, the bounds and
//! the recovered `i` select back to the original range.

use alloc::{boxed::Box, vec::Vec};
use cubecl_ir::{
    NamedRewrite,
    attributes::IndexAttr,
    dialect::{
        branch::{IfOp, RangeLoopOp, YieldOp},
        cmp::{IEqualOp, UGreaterThanOrEqualOp, ULessThanOp, UMaxOp, UMinOp},
        general::{BoolAndOp, BoolOrOp, SelectOp},
        math::{IAddOp, IMulOp, ISubOp, UDivOp, URemOp},
    },
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::MatchRewritePass,
};
use pliron::{
    attribute::Attribute, basic_block::BasicBlock, builtin::ops::ConstantOp,
    irbuild::inserter::OpInsertionPoint, linked_list::ContainsLinkedList, op::op_cast,
    opts::dce::SideEffects, region::Region,
};

/// See the [module docs](self).
pub type QuotientRangePass = MatchRewritePass<QuotientRange>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct QuotientRange;

/// A [`RangeLoopOp`] whose every effect sits under `(base - i * s) % d == 0`.
#[derive(Clone, Copy)]
struct Guarded {
    loop_op: RangeLoopOp,
    base: Value,
    /// Multiplier on the induction variable.
    s: Value,
    /// Divisor whose holes the loop is spinning through.
    d: Value,
}

fn const_index(ctx: &Context, value: Value) -> Option<usize> {
    let attr = value
        .defining_op()?
        .as_op::<ConstantOp>(ctx)?
        .get_value(ctx);
    Some((&*attr as &dyn Attribute).downcast_ref::<IndexAttr>()?.0)
}

/// Build an op, append it at the rewriter's point, and hand back its result.
macro_rules! emit {
    ($ctx:expr, $rewriter:expr, $op:expr) => {{
        let built = $op;
        $rewriter.append_op($ctx, &built);
        built.get_operation().deref($ctx).get_result(0)
    }};
}

/// `base - bound * s`, saturating at zero the way an unsigned numerator must.
fn saturating_sub(
    ctx: &mut Context,
    rw: &mut MatchRewriter,
    base: Value,
    bound: Value,
    s: Value,
) -> Value {
    let scaled = emit!(ctx, rw, IMulOp::new(ctx, bound, s));
    let clamped = emit!(ctx, rw, UMinOp::new(ctx, base, scaled));
    emit!(ctx, rw, ISubOp::new(ctx, base, clamped))
}

/// Match `(base - iv * s) % d == 0`, the shape that skips the holes.
fn as_divisibility(ctx: &Context, cond: Value, iv: Value) -> Option<(Value, Value, Value)> {
    let eq = cond.defining_op()?.as_op::<IEqualOp>(ctx)?;
    let (lhs, rhs) = (eq.lhs(ctx), eq.rhs(ctx));
    let rem = match (const_index(ctx, lhs), const_index(ctx, rhs)) {
        (_, Some(0)) => lhs,
        (Some(0), _) => rhs,
        _ => return None,
    };
    let rem = rem.defining_op()?.as_op::<URemOp>(ctx)?;
    let sub = rem.lhs(ctx).defining_op()?.as_op::<ISubOp>(ctx)?;
    let mul = sub.rhs(ctx).defining_op()?.as_op::<IMulOp>(ctx)?;
    let s = match (mul.lhs(ctx), mul.rhs(ctx)) {
        (l, r) if l == iv => r,
        (l, r) if r == iv => l,
        _ => return None,
    };
    Some((sub.lhs(ctx), s, rem.rhs(ctx)))
}

/// The innermost `if` under which the loop hides its holes, searched through
/// the `then` arms the guard chain nests it in.
fn find_gate(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    iv: Value,
) -> Option<(IfOp, Value, Value, Value)> {
    block.deref(ctx).iter(ctx).find_map(|op| {
        let if_op = op.as_op::<IfOp>(ctx)?;
        match as_divisibility(ctx, if_op.condition(ctx), iv) {
            Some((base, s, d)) => Some((if_op, base, s, d)),
            None => find_gate(ctx, if_op.then_block(ctx), iv),
        }
    })
}

fn has_effects(ctx: &Context, op: Ptr<Operation>) -> bool {
    op_cast::<dyn SideEffects>(&*op.dyn_op(ctx)).is_none_or(|it| it.has_side_effects(ctx))
}

/// Whether every effect reachable from `block` is inside the `then` arm of
/// `gate`. Only `if`s are descended into, so any other region holder is taken
/// to be effectful.
fn effects_only_under(ctx: &Context, block: Ptr<BasicBlock>, gate: IfOp) -> bool {
    block.deref(ctx).iter(ctx).all(|op| {
        if op == gate.get_operation() {
            // Only the iterations that enter the gate survive the rewrite, so
            // the arm its skipped ones take has to be effect free as well.
            effects_only_under(ctx, gate.else_block(ctx), gate)
        } else if op.is_terminator(ctx) {
            true
        } else if let Some(nested) = op.as_op::<IfOp>(ctx) {
            [nested.then_block(ctx), nested.else_block(ctx)]
                .into_iter()
                .all(|block| effects_only_under(ctx, block, gate))
        } else {
            !has_effects(ctx, op)
        }
    })
}

/// Whether `value` is defined outside `region`, and so holds still across the loop.
fn invariant(ctx: &Context, value: Value, region: Ptr<Region>) -> bool {
    let mut block = match value.defining_op() {
        Some(op) => op.deref(ctx).get_parent_block(),
        None => value.defining_block(),
    };
    while let Some(current) = block {
        let Some(parent) = current.deref(ctx).get_parent_region() else {
            return true;
        };
        if parent == region {
            return false;
        }
        block = parent.deref(ctx).get_parent_block(ctx);
    }
    true
}

fn matched(ctx: &Context, op: Ptr<Operation>) -> Option<Guarded> {
    let loop_op = op.as_op::<RangeLoopOp>(ctx)?;
    // A recovered `i` only lands on iterations the original loop had.
    if const_index(ctx, loop_op.step(ctx)) != Some(1) {
        return None;
    }
    let body = loop_op.loop_body(ctx);
    let (gate, base, s, d) = find_gate(ctx, body, loop_op.iter_var(ctx))?;
    let region = loop_op.loop_region(ctx);
    let stable = [base, s, d]
        .into_iter()
        .all(|it| invariant(ctx, it, region));
    // Skipping an iteration is only sound if it could not have done anything.
    (stable && effects_only_under(ctx, body, gate)).then_some(Guarded {
        loop_op,
        base,
        s,
        d,
    })
}

impl MatchRewrite for QuotientRange {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        matched(ctx, op).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some(Guarded {
            loop_op,
            base,
            s,
            d,
        }) = matched(ctx, op)
        else {
            return Ok(());
        };
        let (lo, hi) = (loop_op.start(ctx), loop_op.end(ctx));

        // Everything up to the new loop is loop invariant, so it costs one
        // evaluation per thread rather than one per tap.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));

        let zero = emit!(
            ctx,
            rewriter,
            ConstantOp::new(ctx, Box::new(IndexAttr::new(0)))
        );
        let one = emit!(
            ctx,
            rewriter,
            ConstantOp::new(ctx, Box::new(IndexAttr::new(1)))
        );

        // `s == 0` collapses every `i` onto one `q` and `d == 0` divides
        // nothing; both fall back to the range the loop already had. The
        // clamped divisors keep the bounds below well defined meanwhile.
        let s_zero = emit!(ctx, rewriter, IEqualOp::new(ctx, s, zero));
        let d_zero = emit!(ctx, rewriter, IEqualOp::new(ctx, d, zero));
        let zeroed = emit!(ctx, rewriter, BoolOrOp::new(ctx, s_zero, d_zero));
        let d_safe = emit!(ctx, rewriter, UMaxOp::new(ctx, d, one));
        let s_safe = emit!(ctx, rewriter, UMaxOp::new(ctx, s, one));

        // The quotient reads `base - i * s` as a number, so an `i` that wraps
        // it is one the new bounds cannot reach, and the loop must not have
        // had any. `(hi - 1) * s <= base` says so, divided through by `s` to
        // keep the product itself from overflowing. An empty `hi == 0` wraps
        // into the fallback, which is the same empty loop.
        let last = emit!(ctx, rewriter, ISubOp::new(ctx, hi, one));
        let reach = emit!(ctx, rewriter, UDivOp::new(ctx, base, s_safe));
        let wraps = emit!(ctx, rewriter, ULessThanOp::new(ctx, reach, last));
        let degenerate = emit!(ctx, rewriter, BoolOrOp::new(ctx, zeroed, wraps));

        // `base - i * s` is widest at `i == lo` and narrowest at `i == hi`, so
        // those two ends bracket every quotient the loop can produce. The
        // subtractions saturate because the numerator is unsigned.
        let widest = saturating_sub(ctx, rewriter, base, lo, s);
        let narrowest = saturating_sub(ctx, rewriter, base, hi, s);

        let q_start = emit!(ctx, rewriter, UDivOp::new(ctx, narrowest, d_safe));
        let q_last = emit!(ctx, rewriter, UDivOp::new(ctx, widest, d_safe));
        let q_end = emit!(ctx, rewriter, IAddOp::new(ctx, q_last, one));

        let start = emit!(ctx, rewriter, SelectOp::new(ctx, degenerate, lo, q_start));
        let end = emit!(ctx, rewriter, SelectOp::new(ctx, degenerate, hi, q_end));

        let new_loop = RangeLoopOp::new(ctx, start, end, one);
        rewriter.append_op(ctx, &new_loop);

        // Recover `i` from `q` and prove the recovery exact and in range. The
        // condition is an `and` chain rather than a bare equality, so this loop
        // does not look like a fresh match for the pass.
        let q = new_loop.iter_var(ctx);
        let body = new_loop.loop_body(ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(body));

        // `i` falls as `q` climbs, so a forward `q` would run the body in
        // reverse. Counting back down from the top of the range restores the
        // order the source loop had. Reflecting through `q_start + q_last`
        // would take one op rather than two, but that sum can overflow where
        // neither difference can.
        let travelled = emit!(ctx, rewriter, ISubOp::new(ctx, q, q_start));
        let descending = emit!(ctx, rewriter, ISubOp::new(ctx, q_last, travelled));
        let offset = emit!(ctx, rewriter, IMulOp::new(ctx, descending, d));
        let reachable = emit!(ctx, rewriter, UGreaterThanOrEqualOp::new(ctx, base, offset));
        let scaled = emit!(ctx, rewriter, ISubOp::new(ctx, base, offset));
        let remainder = emit!(ctx, rewriter, URemOp::new(ctx, scaled, s_safe));
        let exact = emit!(ctx, rewriter, IEqualOp::new(ctx, remainder, zero));
        let recovered = emit!(ctx, rewriter, UDivOp::new(ctx, scaled, s_safe));

        let at_least_lo = emit!(
            ctx,
            rewriter,
            UGreaterThanOrEqualOp::new(ctx, recovered, lo)
        );
        let below_hi = emit!(ctx, rewriter, ULessThanOp::new(ctx, recovered, hi));
        let in_range = emit!(ctx, rewriter, BoolAndOp::new(ctx, at_least_lo, below_hi));
        let sound = emit!(ctx, rewriter, BoolAndOp::new(ctx, reachable, exact));
        let sound = emit!(ctx, rewriter, BoolAndOp::new(ctx, sound, in_range));
        let guard = emit!(ctx, rewriter, BoolOrOp::new(ctx, degenerate, sound));
        let iter_var = emit!(ctx, rewriter, SelectOp::new(ctx, degenerate, q, recovered));

        // The gate ends the loop body, and both of its arms are empty until the
        // original body moves in below.
        let gate = IfOp::new(ctx, guard);
        rewriter.append_op(ctx, &gate);
        for block in [body, gate.then_block(ctx), gate.else_block(ctx)] {
            let terminator = YieldOp::new(ctx);
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(block));
            rewriter.append_op(ctx, &terminator);
        }

        // The body keeps its own guards; against the recovered `i` they are
        // redundant, and left for the later simplification passes.
        let old_body = loop_op.loop_body(ctx);
        let mut point = OpInsertionPoint::AtBlockStart(gate.then_block(ctx));
        let old_ops = old_body.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for old in old_ops {
            if old.is_terminator(ctx) {
                continue;
            }
            rewriter.move_operation(ctx, old, point);
            point = OpInsertionPoint::AfterOperation(old);
        }
        rewriter.replace_value_uses_with(ctx, loop_op.iter_var(ctx), iter_var);
        rewriter.erase_operation(ctx, op);

        Ok(())
    }
}
