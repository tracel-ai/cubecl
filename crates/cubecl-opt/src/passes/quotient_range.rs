//! Re-index a divisibility-guarded range loop by the quotient of its guard.
//!
//! Experimental: backend pipelines leave this pass disabled unless
//! `CUBECL_ENABLE_QUOTIENT_RANGE=1` is set before compilation. A trip-count
//! heuristic cannot guarantee that changing a kernel makes it faster on every
//! device. Keeping the pass out of the default pipeline preserves its input IR
//! without adding runtime guards or a second loop. Directly running this
//! rewrite is also an explicit opt-in.
//!
//! ```text
//! for i in lo..hi {
//!     if i * s <= base {                       // reachability guard
//!         if (base - i * s) % d == 0 { body }  // divisibility gate
//!     }
//! }
//! ```
//!
//! runs `hi - lo` times but only reaches `body` on the `i` where `d` divides
//! `base - i * s`. Dilated conv-transpose is the motivating shape: the window
//! it walks is `kernel * dilation` wide and every `dilation`-th tap survives,
//! so the trip count grows with `dilation` while the arithmetic does not. The
//! reachability guard is what makes an `i` past `base` effect-free — the
//! rewrite drops those iterations instead of restoring them.
//!
//! The surviving taps are in bijection with the quotient
//! `q = (base - i * s) / d`, whose inverse is `i = (base - q * d) / s` wherever
//! `s` divides `base - q * d`. Driving the loop by `q` reaches each tap exactly
//! once. `q` climbs as `i` falls, so the loop counts back down to restore the
//! source order.
//!
//! Once enabled, the rewrite uses a cost heuristic. The chain costs ops per
//! surviving tap whether or not the loop had holes, so it fires only where the
//! loop skips at least as many iterations as it keeps. Constant operands get
//! that question answered at compile time — a window that would never pay is
//! left alone — and dynamic ones are wrapped in a runtime unswitch on
//! [`quotient_applicable`], which also keeps the arithmetic well defined and
//! parks losing windows in the source arm. In the paying arm both guards are
//! provable on every surviving tap, so the body moves across with them
//! stripped, along with the products the recovery already computed.
//!
//! The arithmetic is written as polyfill functions: [`quotient_applicable`] is
//! the unswitch condition, [`quotient_bounds`] hoists the quotient range, and
//! [`recover_tap`] recovers `i` and the tap index from `q`. The pass itself
//! only matches the guarded-loop shape, clones the body, and parks the source
//! loop in the unswitch's else arm.

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

/// A [`RangeLoopOp`] whose every effect sits under
/// `i * s <= base && (base - i * s) % d == 0`.
#[derive(Clone, Copy)]
struct Guarded {
    loop_op: RangeLoopOp,
    /// The `i * s <= base` guard: iterations past `base` cannot reach the body.
    outer: IfOp,
    /// The `(base - i * s) % d == 0` gate the holes hide behind.
    gate: IfOp,
    /// `i * s`: the reachability guard's product, the body's `numerator_tmp`.
    scaled: Value,
    /// `base - i * s`: the dividend the body divides by `d`.
    numerator: Value,
    /// `(base - i * s) % d`: zero on every tap the body runs for.
    rem: Value,
    /// The zero the gate compares the remainder against.
    zero: Value,
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

/// Match `(base - iv * s) % d == 0`, the shape that skips the holes. Returns
/// `(base, s, d)`, the dividend `base - iv * s`, the remainder the gate
/// compares against zero, and that zero.
fn as_divisibility(
    ctx: &Context,
    cond: Value,
    iv: Value,
) -> Option<(Value, Value, Value, Value, Value, Value)> {
    let eq = cond.defining_op()?.as_op::<IEqualOp>(ctx)?;
    let (lhs, rhs) = (eq.lhs(ctx), eq.rhs(ctx));
    let (rem, zero) = match (const_index(ctx, lhs), const_index(ctx, rhs)) {
        (_, Some(0)) => (lhs, rhs),
        (Some(0), _) => (rhs, lhs),
        _ => return None,
    };
    let rem_op = rem.defining_op()?.as_op::<URemOp>(ctx)?;
    let sub = rem_op.lhs(ctx).defining_op()?.as_op::<ISubOp>(ctx)?;
    let mul = sub.rhs(ctx).defining_op()?.as_op::<IMulOp>(ctx)?;
    let s = match (mul.lhs(ctx), mul.rhs(ctx)) {
        (l, r) if l == iv => r,
        (l, r) if r == iv => l,
        _ => return None,
    };
    Some((sub.lhs(ctx), s, rem_op.rhs(ctx), rem_op.lhs(ctx), rem, zero))
}

/// Match `base >= iv * s` (or its swap): the guard that makes an `i` past
/// `base` effect-free. Returns `(base, s, iv * s)`.
fn as_reachability(ctx: &Context, cond: Value, iv: Value) -> Option<(Value, Value, Value)> {
    let defining = cond.defining_op()?;
    let (prod, base) = if let Some(cmp) = defining.as_op::<UGreaterThanOrEqualOp>(ctx) {
        (cmp.rhs(ctx), cmp.lhs(ctx))
    } else if let Some(cmp) = defining.as_op::<ULessThanOrEqualOp>(ctx) {
        (cmp.lhs(ctx), cmp.rhs(ctx))
    } else {
        return None;
    };
    let mul = prod.defining_op()?.as_op::<IMulOp>(ctx)?;
    let s = match (mul.lhs(ctx), mul.rhs(ctx)) {
        (l, r) if l == iv => r,
        (l, r) if r == iv => l,
        _ => return None,
    };
    Some((base, s, prod))
}

/// The `i * s <= base` guard, searched through the `then` arms.
fn find_reachability(ctx: &Context, block: Ptr<BasicBlock>, iv: Value) -> Option<IfOp> {
    block.deref(ctx).iter(ctx).find_map(|op| {
        let if_op = op.as_op::<IfOp>(ctx)?;
        match as_reachability(ctx, if_op.condition(ctx), iv) {
            Some(_) => Some(if_op),
            None => find_reachability(ctx, if_op.then_block(ctx), iv),
        }
    })
}

/// The innermost divisibility `if` under `block`, searched through the `then`
/// arms the guard chain nests it in.
fn find_gate(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    iv: Value,
) -> Option<(IfOp, Value, Value, Value, Value, Value, Value)> {
    block.deref(ctx).iter(ctx).find_map(|op| {
        let if_op = op.as_op::<IfOp>(ctx)?;
        match as_divisibility(ctx, if_op.condition(ctx), iv) {
            Some((base, s, d, numerator, rem, zero)) => {
                Some((if_op, base, s, d, numerator, rem, zero))
            }
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
    let outer = find_reachability(ctx, body, iv)?;
    let (base, s, scaled) = as_reachability(ctx, outer.condition(ctx), iv)?;
    let (gate, gate_base, gate_s, d, numerator, rem, zero) =
        find_gate(ctx, outer.then_block(ctx), iv)?;
    if base != gate_base || s != gate_s {
        return None;
    }
    // d <= 1 has no holes. At d == 2 the recovery chain's cost still outweighs
    // the saved iterations in small-loop benchmarks, so keep the source loop.
    if matches!(const_index(ctx, d), Some(0..=2)) {
        return None;
    }
    // With constant operands the profitability question is settled here: an
    // unswitch whose condition is statically false would only ever route to
    // the source arm, so the rewrite declines instead of emitting it. Dynamic
    // windows — a clipped dilated window has far fewer holes than its nominal
    // `d` suggests — keep the runtime check in `quotient_applicable`.
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
        scaled,
        numerator,
        rem,
        zero,
        base,
        s,
        d,
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

/// The rewrite fires where skipping beats checking. The chain costs its ops
/// per surviving tap whether or not the loop had holes, so the rewrite is only
/// worth it when the loop skips at least as many iterations as it keeps — a
/// per-window decision, since a clipped window of a dilated loop can have far
/// fewer holes than its nominal `d` suggests. The `s`/`d` checks keep the
/// arithmetic well defined and park the pathological operands on the source
/// path.
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

/// The quotient range bracketing every tap the loop can produce. Hoisted out
/// of the loop: one evaluation per loop entry, not one per tap.
///
/// `base - i * s` is widest at `i == lo` and narrowest at `i == hi`, so those
/// two ends bracket every quotient. The subtractions saturate because the
/// numerator is unsigned. The range is a superset of the surviving taps;
/// [`recover_tap`] re-validates each one. The divisors are clamped so the
/// range stays defined even on operands [`quotient_applicable`] rejects.
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

/// Recover the source iteration `i` from `q`. Bounds guarantee `tap*d <= base`;
/// `exact` keeps `i` integral and `in_range` keeps it in the source loop.
///
/// `i` falls as `q` climbs, so a forward `q` would run the body in reverse.
/// Counting back down from the top of the range (`tap`) restores the source
/// order. `tap * d` is exactly `base - i * s` — the numerator the body divides
/// by `d` to find its kernel position — and `scaled` is `i * s`; the stripped
/// body substitutes both, so its own recomputation disappears.
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

/// Untyped [`quotient_applicable`] expansion: binds `Idx`/`Lane` to the
/// loop's index values and hands the result back as a [`Value`].
fn expand_applicable(
    scope: &Scope,
    lo: Value,
    hi: Value,
    s: Value,
    d: Value,
    q_start: Value,
    q_end: Value,
) -> Value {
    scope.register_value_type::<Idx, Lane>(s);
    quotient_applicable::expand::<Idx, Lane>(
        scope,
        lo.into(),
        hi.into(),
        s.into(),
        d.into(),
        q_start.into(),
        q_end.into(),
        (u64::MAX >> (64 - scope.ctx().address_type().size_bits())) as i64,
    )
    .value(scope)
}

/// Untyped [`quotient_bounds`] expansion; see [`expand_applicable`].
fn expand_bounds(
    scope: &Scope,
    lo: Value,
    hi: Value,
    base: Value,
    s: Value,
    d: Value,
) -> (Value, Value, Value) {
    scope.register_value_type::<Idx, Lane>(lo);
    let (q_start, q_last, q_end) = quotient_bounds::expand::<Idx, Lane>(
        scope,
        lo.into(),
        hi.into(),
        base.into(),
        s.into(),
        d.into(),
    );
    (
        q_start.value(scope),
        q_last.value(scope),
        q_end.value(scope),
    )
}

/// Untyped [`recover_tap`] expansion; see [`expand_applicable`].
#[allow(clippy::too_many_arguments)]
fn expand_recover(
    scope: &Scope,
    q: Value,
    lo: Value,
    hi: Value,
    base: Value,
    s: Value,
    d: Value,
    q_start: Value,
    q_last: Value,
    unit_stride: bool,
) -> (Value, Value, Value, Value, Value) {
    scope.register_value_type::<Idx, Lane>(q);
    let (i, tap, numerator, scaled, keep) = recover_tap::expand::<Idx, Lane>(
        scope,
        q.into(),
        lo.into(),
        hi.into(),
        base.into(),
        s.into(),
        d.into(),
        q_start.into(),
        q_last.into(),
        unit_stride,
    );
    (
        i.value(scope),
        tap.value(scope),
        numerator.value(scope),
        scaled.value(scope),
        keep.value(scope),
    )
}

/// The body's `kernel_y = numerator / d`: the tap index the chain recovered.
fn as_quotient_of(ctx: &Context, op: Ptr<Operation>, numerator: Value, d: Value) -> Option<Value> {
    let udiv = op.as_op::<UDivOp>(ctx)?;
    (udiv.lhs(ctx) == numerator && udiv.rhs(ctx) == d).then(|| op.deref(ctx).get_result(0))
}

/// Clone the body into the stripped gate: flatten the two guards the chain now
/// proves, drop the defs it has already substituted, and copy the rest.
/// Returns true when a copied terminator ends the destination block.
fn clone_body(
    ctx: &mut Context,
    rewriter: &mut MatchRewriter,
    mapper: &mut IrMapping,
    src: Ptr<BasicBlock>,
    dst: Ptr<BasicBlock>,
    outer: IfOp,
    gate: IfOp,
    subbed: &[Value],
    numerator: Value,
    d: Value,
    tap: Value,
) -> bool {
    let ops = src.deref(ctx).iter(ctx).collect::<Vec<_>>();
    for old in ops {
        if old.as_op::<YieldOp>(ctx).is_some() {
            continue;
        }
        if old == outer.get_operation() || old == gate.get_operation() {
            // The chain proves reachability and divisibility on every tap the
            // body runs for, so both guards are free to go.
            let then = if old == outer.get_operation() {
                outer.then_block(ctx)
            } else {
                gate.then_block(ctx)
            };
            if clone_body(
                ctx, rewriter, mapper, then, dst, outer, gate, subbed, numerator, d, tap,
            ) {
                return true;
            }
            continue;
        }
        let result = {
            let op_ref = old.deref(ctx);
            (op_ref.result_types().count() == 1).then(|| op_ref.get_result(0))
        };
        if let Some(result) = result {
            if subbed.contains(&result) {
                continue;
            }
            if let Some(quotient) = as_quotient_of(ctx, old, numerator, d) {
                mapper.map_value(quotient, tap);
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
            outer,
            gate,
            scaled,
            numerator,
            rem,
            zero,
            base,
            s,
            d,
        }) = matched(ctx, op)
        else {
            return Ok(());
        };
        let (lo, hi) = (loop_op.start(ctx), loop_op.end(ctx));
        let iv = loop_op.iter_var(ctx);

        // The body may use the remainder itself. Its replacement zero must
        // dominate both arms, even when originally declared inside the loop.
        if !invariant(ctx, zero, loop_op.loop_region(ctx)) {
            rewriter.move_operation(
                ctx,
                zero.defining_op().unwrap(),
                OpInsertionPoint::BeforeOperation(op),
            );
        }

        // The source loop parks in the unswitch's else arm with its shape
        // intact; mark it so the fixpoint walk leaves it there.
        mark_unswitched(ctx, op);

        // Everything up to the unswitch is loop invariant, so it costs one
        // evaluation per loop entry rather than one per tap. The quotient range
        // also feeds the trip-count arithmetic the profitability check
        // compares against, so both are computed up front.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        let (q_start, q_last, q_end) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            expand_bounds(&scope, lo, hi, base, s, d)
        };
        let applicable = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            expand_applicable(&scope, lo, hi, s, d, q_start, q_end)
        };
        let unswitch = IfOp::new(ctx, applicable);
        rewriter.append_op(ctx, &unswitch);
        let (then_block, else_block) = (unswitch.then_block(ctx), unswitch.else_block(ctx));

        // The quotient arm.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(then_block));
        let q_loop = RangeLoopOp::new(ctx, q_start, q_end, loop_op.step(ctx));
        mark_unswitched(ctx, q_loop.get_operation());
        rewriter.append_op(ctx, &q_loop);
        let q = q_loop.iter_var(ctx);
        let q_body = q_loop.loop_body(ctx);

        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(q_body));
        let unit_stride = const_index(ctx, s) == Some(1);
        let (iter_var, tap, num, scaled_v, keep) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            expand_recover(&scope, q, lo, hi, base, s, d, q_start, q_last, unit_stride)
        };

        let body_gate = IfOp::new(ctx, keep);
        rewriter.append_op(ctx, &body_gate);

        // The body keeps neither guard and none of the arithmetic the chain
        // already did. What the guards computed is known on every tap the body
        // runs for: the remainder was zero, the reachability and divisibility
        // checks held — the same thing `keep` says. Substituting them means
        // neither the guards nor their recomputation survive the move.
        let mut mapper = IrMapping::new();
        mapper.map_value(iv, iter_var);
        mapper.map_value(scaled, scaled_v);
        mapper.map_value(numerator, num);
        mapper.map_value(rem, zero);
        mapper.map_value(outer.condition(ctx), keep);
        mapper.map_value(gate.condition(ctx), keep);
        let subbed = [
            scaled,
            numerator,
            rem,
            outer.condition(ctx),
            gate.condition(ctx),
        ];
        let body_exits = clone_body(
            ctx,
            rewriter,
            &mut mapper,
            loop_op.loop_body(ctx),
            body_gate.then_block(ctx),
            outer,
            gate,
            &subbed,
            numerator,
            d,
            tap,
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

        // The source arm.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(else_block));
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockStart(else_block));

        Ok(())
    }
}
