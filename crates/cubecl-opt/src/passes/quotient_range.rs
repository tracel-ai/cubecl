//! Re-index loops whose effects are guarded by `i*s <= base && (base-i*s)%d == 0`.
//!
//! ```text
//! for i in lo..hi {
//!     if i * s <= base {
//!         if (base - i * s) % d == 0 { body }
//!     }
//! }
//! ```
//!
//! Enumerating `q = (base-i*s)/d` skips the holes in dilated conv-transpose.
//! Descending quotients preserve source order; recovery validates each tap.
//! Unsafe or unprofitable dynamic windows use the original loop. The arithmetic
//! lives in three polyfills; the rewrite substitutes their results into the body.
//! Backends require [`enabled`] to opt in; directly running the rewrite is also
//! explicit opt-in.

use alloc::vec::Vec;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_ir::{
    NamedRewrite,
    attributes::IndexAttr,
    dialect::{
        branch::{IfOp, RangeLoopOp, YieldOp},
        cmp::{IEqualOp, UGreaterThanOrEqualOp, ULessThanOrEqualOp},
        math::{IMulOp, ISubOp, UDivOp, URemOp},
    },
    ident,
    prelude::{Inserter as _, Rewriter as _, *},
    rewrite::MatchRewritePass,
};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::{attributes::UnitAttr, ops::ConstantOp},
    irbuild::{
        cloning::{IrMapping, clone_operation},
        inserter::OpInsertionPoint,
    },
    linked_list::ContainsLinkedList,
    op::op_cast,
    opts::dce::SideEffects,
    region::Region,
};

/// See the [module docs](self).
pub type QuotientRangePass = MatchRewritePass<QuotientRange>;

#[derive(Default, Clone, Copy, NamedRewrite)]
pub struct QuotientRange;

define_scalar!(Idx);
define_size!(Lane);

/// Marker on a loop that already lives in an unswitch this pass created, so
/// the fixpoint walk does not unswitch it again.
const UNSWITCHED: &str = "quotient_range_unswitched";

/// The two guards and the affine expression they share.
#[derive(Clone, Copy)]
struct Guarded {
    loop_op: RangeLoopOp,
    outer: IfOp,
    gate: IfOp,
    product: Value,
    division: Divisibility,
}

#[derive(Clone, Copy)]
struct Reachability {
    base: Value,
    scale: Value,
    product: Value,
}

#[derive(Clone, Copy)]
struct Divisibility {
    base: Value,
    scale: Value,
    divisor: Value,
    numerator: Value,
    remainder: Value,
    zero: Value,
}

struct PreparedRange {
    start: Value,
    last: Value,
    end: Value,
    applicable: Value,
}

struct RecoveredTap {
    index: Value,
    quotient: Value,
    numerator: Value,
    scaled: Value,
    keep: Value,
}

fn const_index(ctx: &Context, value: Value) -> Option<usize> {
    let attr = value
        .defining_op()?
        .as_op::<ConstantOp>(ctx)?
        .get_value(ctx);
    Some((&*attr as &dyn Attribute).downcast_ref::<IndexAttr>()?.0)
}

/// Decline statically unsafe or unprofitable windows without emitting an unswitch.
fn holes_pay(ctx: &Context, lo: usize, hi: usize, base: usize, s: usize, d: usize) -> bool {
    let max_index = u64::MAX >> (64 - ctx.address_type().size_bits());
    let (lo, hi, base, s, d) = (lo as u64, hi as u64, base as u64, s as u64, d as u64);
    if lo >= hi || s == 0 || d < 3 || hi > max_index / s || base > max_index {
        return false;
    }
    let q_start = base.saturating_sub(hi * s) / d + u64::from(base >= hi * s);
    let q_last = base.saturating_sub(lo * s) / d;
    q_last + 1 - q_start <= (hi - lo) / 2
}

fn scale_of(ctx: &Context, product: Value, iv: Value) -> Option<Value> {
    let mul = product.defining_op()?.as_op::<IMulOp>(ctx)?;
    match (mul.lhs(ctx), mul.rhs(ctx)) {
        (l, r) if l == iv => Some(r),
        (l, r) if r == iv => Some(l),
        _ => None,
    }
}

/// Match `(base - iv * scale) % divisor == 0`.
fn as_divisibility(ctx: &Context, cond: Value, iv: Value) -> Option<Divisibility> {
    let eq = cond.defining_op()?.as_op::<IEqualOp>(ctx)?;
    let (lhs, rhs) = (eq.lhs(ctx), eq.rhs(ctx));
    let (rem, zero) = match (const_index(ctx, lhs), const_index(ctx, rhs)) {
        (_, Some(0)) => (lhs, rhs),
        (Some(0), _) => (rhs, lhs),
        _ => return None,
    };
    let rem_op = rem.defining_op()?.as_op::<URemOp>(ctx)?;
    let sub = rem_op.lhs(ctx).defining_op()?.as_op::<ISubOp>(ctx)?;
    Some(Divisibility {
        base: sub.lhs(ctx),
        scale: scale_of(ctx, sub.rhs(ctx), iv)?,
        divisor: rem_op.rhs(ctx),
        numerator: rem_op.lhs(ctx),
        remainder: rem,
        zero,
    })
}

/// Match `base >= iv * scale` or its swapped comparison.
fn as_reachability(ctx: &Context, cond: Value, iv: Value) -> Option<Reachability> {
    let defining = cond.defining_op()?;
    let (prod, base) = if let Some(cmp) = defining.as_op::<UGreaterThanOrEqualOp>(ctx) {
        (cmp.rhs(ctx), cmp.lhs(ctx))
    } else if let Some(cmp) = defining.as_op::<ULessThanOrEqualOp>(ctx) {
        (cmp.lhs(ctx), cmp.rhs(ctx))
    } else {
        return None;
    };
    Some(Reachability {
        base,
        scale: scale_of(ctx, prod, iv)?,
        product: prod,
    })
}

/// Depth-first search through `then` arms, stopping at the first matching guard.
fn find_guard<T>(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    match_condition: &impl Fn(Value) -> Option<T>,
) -> Option<(IfOp, T)> {
    block.deref(ctx).iter(ctx).find_map(|op| {
        let if_op = op.as_op::<IfOp>(ctx)?;
        match_condition(if_op.condition(ctx))
            .map(|matched| (if_op, matched))
            .or_else(|| find_guard(ctx, if_op.then_block(ctx), match_condition))
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
            // Returning from a skipped iteration is observable too.
            op.as_op::<YieldOp>(ctx).is_some()
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

fn marked_unswitched(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.deref(ctx)
        .attributes
        .get::<UnitAttr>(&ident(UNSWITCHED))
        .is_some()
}

fn mark_unswitched(ctx: &mut Context, op: Ptr<Operation>) {
    op.deref_mut(ctx)
        .attributes
        .set(ident(UNSWITCHED), UnitAttr::new());
}

/// Whether backend pipelines should register the experimental pass.
///
/// Disabled by default, including without `std`. Set
/// `CUBECL_ENABLE_QUOTIENT_RANGE=1` before compilation to opt in. The existing
/// `CUBECL_DISABLE_QUOTIENT_RANGE` override takes precedence. Read once per
/// process, outside the IR matcher and device execution path.
pub fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            experimental_requested(
                std::env::var("CUBECL_ENABLE_QUOTIENT_RANGE")
                    .ok()
                    .as_deref(),
                std::env::var_os("CUBECL_DISABLE_QUOTIENT_RANGE").is_some(),
            )
        })
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

#[cfg(any(feature = "std", test))]
fn experimental_requested(enable: Option<&str>, disable: bool) -> bool {
    enable == Some("1") && !disable
}

fn matched(ctx: &Context, op: Ptr<Operation>) -> Option<Guarded> {
    if marked_unswitched(ctx, op) {
        return None;
    }
    let loop_op = op.as_op::<RangeLoopOp>(ctx)?;
    // A recovered `i` only lands on iterations the original loop had.
    if const_index(ctx, loop_op.step(ctx)) != Some(1) {
        return None;
    }
    let body = loop_op.loop_body(ctx);
    let iv = loop_op.iter_var(ctx);
    // Reachability excludes subtraction underflow. Multiplication overflow
    // needs a separate check in quotient_applicable.
    let (outer, reach) = find_guard(ctx, body, &|cond| as_reachability(ctx, cond, iv))?;
    let (gate, division) = find_guard(ctx, outer.then_block(ctx), &|cond| {
        as_divisibility(ctx, cond, iv)
    })?;
    if (reach.base, reach.scale) != (division.base, division.scale) {
        return None;
    }
    let (base, s, d) = (division.base, division.scale, division.divisor);
    // d <= 1 has no holes. At d == 2 the recovery chain's cost still outweighs
    // the saved iterations in small-loop benchmarks, so keep the source loop.
    if matches!(const_index(ctx, d), Some(0..=2)) {
        return None;
    }
    // Static rejection avoids generating a runtime unswitch for losing windows.
    if let (Some(lo), Some(hi), Some(b), Some(s_c), Some(d_c)) = (
        const_index(ctx, loop_op.start(ctx)),
        const_index(ctx, loop_op.end(ctx)),
        const_index(ctx, base),
        const_index(ctx, s),
        const_index(ctx, d),
    ) && !holes_pay(ctx, lo, hi, b, s_c, d_c)
    {
        return None;
    }
    let region = loop_op.loop_region(ctx);
    let stable = [base, s, d]
        .into_iter()
        .all(|it| invariant(ctx, it, region));
    // Skipping an iteration is only sound if it could not have done anything.
    (stable && effects_only_under(ctx, body, gate)).then_some(Guarded {
        loop_op,
        outer,
        gate,
        product: reach.product,
        division,
    })
}

#[cfg(test)]
mod policy_tests {
    use super::experimental_requested;

    #[test]
    fn requires_explicit_opt_in() {
        for value in [None, Some(""), Some("0"), Some("true")] {
            assert!(!experimental_requested(value, false));
        }
        assert!(experimental_requested(Some("1"), false));
        assert!(!experimental_requested(Some("1"), true));
    }
}

/// Require at least as many skipped iterations as candidates. Invalid operands
/// and overflowing products take the source arm, including for clipped windows.
#[cube]
fn quotient_applicable<I: Int, N: Size>(
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
    q_start: Vector<I, N>,
    q_end: Vector<I, N>,
    #[comptime] max_index: i64,
) -> Vector<bool, N> {
    let taps = q_end - q_start;
    let trips = hi - lo;
    let s_safe = s.max(Vector::new(I::new(1)));
    // Check the exclusive endpoint too: quotient_bounds multiplies hi by s.
    let no_overflow = hi.less_equal(&(Vector::new(I::new(max_index)) / s_safe));
    s.greater_equal(&Vector::new(I::new(1)))
        .vec_and(d.greater_equal(&Vector::new(I::new(3))))
        .vec_and(lo.less_than(&hi))
        .vec_and(no_overflow)
        .vec_and(taps.less_equal(&(trips / Vector::new(I::new(2)))))
}

/// Bracket all quotients using the two endpoints, once per loop entry. Saturate
/// unsigned subtractions and clamp divisors even for rejected windows, since
/// these bounds are evaluated before the unswitch.
#[cube]
fn quotient_bounds<I: Int, N: Size>(
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    base: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
) -> (Vector<I, N>, Vector<I, N>, Vector<I, N>) {
    let s_safe = s.max(Vector::new(I::new(1)));
    // d < 3 always takes the source arm; clamping to 3 also prevents q_last + 1
    // from overflowing while computing speculative bounds for that arm.
    let d_safe = d.max(Vector::new(I::new(3)));
    let widest = base - (lo * s_safe).min(base);
    let upper_product = hi * s_safe;
    let narrowest = base - upper_product.min(base);
    // Exclude q for i == hi: base - q*d must be strictly less than hi*s.
    let past_end = Vector::<I, N>::cast_from(base.greater_equal(&upper_product));
    let q_start = narrowest / d_safe + past_end;
    let q_last = widest / d_safe;
    (q_start, q_last, q_last + Vector::new(I::new(1)))
}

/// Descending quotients restore source order. Bounds guarantee `tap*d <= base`;
/// exactness and range checks validate i. The other results replace arithmetic
/// already present in the body: `tap*d = base-i*s` and `scaled = i*s`.
#[cube]
fn recover_tap<I: Int, N: Size>(
    q: Vector<I, N>,
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    base: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
    q_start: Vector<I, N>,
    q_last: Vector<I, N>,
    #[comptime] unit_stride: bool,
) -> (
    Vector<I, N>,
    Vector<I, N>,
    Vector<I, N>,
    Vector<I, N>,
    Vector<bool, N>,
) {
    let tap = q_last - (q - q_start);
    let numerator = tap * d;
    let scaled = base - numerator;
    let recovered = if unit_stride { scaled } else { scaled / s };
    let exact = if unit_stride {
        Vector::new(true)
    } else {
        (scaled % s).equal(&Vector::new(I::new(0)))
    };
    let in_range = recovered
        .greater_equal(&lo)
        .vec_and(recovered.less_than(&hi));
    let keep = exact.vec_and(in_range);
    (recovered, tap, numerator, scaled, keep)
}

impl Guarded {
    /// Expand both invariant polyfills with one binding of the loop's index type.
    fn prepare(&self, scope: &Scope) -> PreparedRange {
        let lo = self.loop_op.start(scope.ctx());
        let hi = self.loop_op.end(scope.ctx());
        let div = self.division;
        scope.register_value_type::<Idx, Lane>(lo);
        let (start, last, end) = quotient_bounds::expand::<Idx, Lane>(
            scope,
            lo.into(),
            hi.into(),
            div.base.into(),
            div.scale.into(),
            div.divisor.into(),
        );
        let (start, last, end) = (start.value(scope), last.value(scope), end.value(scope));
        let applicable = quotient_applicable::expand::<Idx, Lane>(
            scope,
            lo.into(),
            hi.into(),
            div.scale.into(),
            div.divisor.into(),
            start.into(),
            end.into(),
            (u64::MAX >> (64 - scope.ctx().address_type().size_bits())) as i64,
        )
        .value(scope);
        PreparedRange {
            start,
            last,
            end,
            applicable,
        }
    }

    fn recover(&self, scope: &Scope, q: Value, range: &PreparedRange) -> RecoveredTap {
        let div = self.division;
        let (index, quotient, numerator, scaled, keep) = recover_tap::expand::<Idx, Lane>(
            scope,
            q.into(),
            self.loop_op.start(scope.ctx()).into(),
            self.loop_op.end(scope.ctx()).into(),
            div.base.into(),
            div.scale.into(),
            div.divisor.into(),
            range.start.into(),
            range.last.into(),
            const_index(scope.ctx(), div.scale) == Some(1),
        );
        RecoveredTap {
            index: index.value(scope),
            quotient: quotient.value(scope),
            numerator: numerator.value(scope),
            scaled: scaled.value(scope),
            keep: keep.value(scope),
        }
    }

    /// The same substitutions drive value mapping and omission of redundant definitions.
    fn replacements(&self, ctx: &Context, tap: &RecoveredTap) -> [(Value, Value); 6] {
        [
            (self.loop_op.iter_var(ctx), tap.index),
            (self.product, tap.scaled),
            (self.division.numerator, tap.numerator),
            (self.division.remainder, self.division.zero),
            (self.outer.condition(ctx), tap.keep),
            (self.gate.condition(ctx), tap.keep),
        ]
    }

    /// Flatten the proven guards and substitute recovered values. True means an
    /// early exit terminated the destination; callers must not append a yield.
    fn clone_body(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        mapper: &mut IrMapping,
        src: Ptr<BasicBlock>,
        dst: Ptr<BasicBlock>,
        tap: &RecoveredTap,
    ) -> bool {
        let replacements = self.replacements(ctx, tap);
        let ops = src.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for old in ops {
            if old.as_op::<YieldOp>(ctx).is_some() {
                continue;
            }
            if old == self.outer.get_operation() || old == self.gate.get_operation() {
                let then = old.as_op::<IfOp>(ctx).unwrap().then_block(ctx);
                if self.clone_body(ctx, rewriter, mapper, then, dst, tap) {
                    return true;
                }
                continue;
            }
            let result = {
                let op = old.deref(ctx);
                (op.result_types().count() == 1).then(|| op.get_result(0))
            };
            if let Some(result) = result {
                if replacements.iter().any(|(from, _)| *from == result) {
                    continue;
                }
                if old.as_op::<UDivOp>(ctx).is_some_and(|div| {
                    div.lhs(ctx) == self.division.numerator && div.rhs(ctx) == self.division.divisor
                }) {
                    mapper.map_value(result, tap.quotient);
                    continue;
                }
            }
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(dst));
            let cloned = clone_operation(old, ctx, rewriter, mapper);
            rewriter.append_operation(ctx, cloned);
            if old.is_terminator(ctx) {
                return true;
            }
        }
        false
    }
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
        let Some(guard) = matched(ctx, op) else {
            return Ok(());
        };
        // A locally declared replacement zero must dominate both loop versions.
        let zero = guard.division.zero;
        if !invariant(ctx, zero, guard.loop_op.loop_region(ctx)) {
            rewriter.move_operation(
                ctx,
                zero.defining_op().unwrap(),
                OpInsertionPoint::BeforeOperation(op),
            );
        }
        mark_unswitched(ctx, op);

        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        let range = guard.prepare(&Scope::from_context_and_inserter(ctx, rewriter));
        let unswitch = IfOp::new(ctx, range.applicable);
        rewriter.append_op(ctx, &unswitch);
        let (then_block, else_block) = (unswitch.then_block(ctx), unswitch.else_block(ctx));

        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(then_block));
        let q_loop = RangeLoopOp::new(ctx, range.start, range.end, guard.loop_op.step(ctx));
        mark_unswitched(ctx, q_loop.get_operation());
        rewriter.append_op(ctx, &q_loop);
        let q = q_loop.iter_var(ctx);
        let q_body = q_loop.loop_body(ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(q_body));
        let tap = guard.recover(&Scope::from_context_and_inserter(ctx, rewriter), q, &range);
        let body_gate = IfOp::new(ctx, tap.keep);
        rewriter.append_op(ctx, &body_gate);

        let mut mapper = IrMapping::new();
        for (from, to) in guard.replacements(ctx, &tap) {
            mapper.map_value(from, to);
        }
        let body_exits = guard.clone_body(
            ctx,
            rewriter,
            &mut mapper,
            guard.loop_op.loop_body(ctx),
            body_gate.then_block(ctx),
            &tap,
        );
        for block in [q_body, body_gate.else_block(ctx), then_block, else_block] {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(block));
            let terminator = YieldOp::new(ctx);
            rewriter.append_op(ctx, &terminator);
        }
        if !body_exits {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(body_gate.then_block(ctx)));
            let terminator = YieldOp::new(ctx);
            rewriter.append_op(ctx, &terminator);
        }

        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(else_block));
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockStart(else_block));
        Ok(())
    }
}
