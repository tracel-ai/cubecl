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
//! lives in polyfills; cheap rejections bypass bound preparation, which is moved
//! across immediately enclosing range loops when all its inputs are invariant.
//! The rewrite substitutes recovered values into the body.

use alloc::{boxed::Box, vec::Vec};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_ir::{
    NamedRewrite,
    attributes::{BoolAttr, IndexAttr},
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

#[derive(CubeType)]
#[allow(dead_code)] // Fields are read through the generated expansion type.
struct QuotientRangeValues<I: Int, N: Size> {
    start: Vector<I, N>,
    last: Vector<I, N>,
    end: Vector<I, N>,
    applicable: Vector<bool, N>,
}

#[derive(CubeType)]
#[allow(dead_code)] // Fields are read through the generated expansion type.
struct RecoveredIndex<I: Int, N: Size> {
    index: Vector<I, N>,
    quotient: Vector<I, N>,
    numerator: Vector<I, N>,
    scaled: Vector<I, N>,
    keep: Vector<bool, N>,
}

fn const_index(ctx: &Context, value: Value) -> Option<usize> {
    let attr = value
        .defining_op()?
        .as_op::<ConstantOp>(ctx)?
        .get_value(ctx);
    Some((&*attr as &dyn Attribute).downcast_ref::<IndexAttr>()?.0)
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
    // needs a separate check in prepare_range.
    let (outer, reach) = find_guard(ctx, body, &|cond| as_reachability(ctx, cond, iv))?;
    let (gate, division) = find_guard(ctx, outer.then_block(ctx), &|cond| {
        as_divisibility(ctx, cond, iv)
    })?;
    if (reach.base, reach.scale) != (division.base, division.scale) {
        return None;
    }
    let (base, s, d) = (division.base, division.scale, division.divisor);
    // Apply the small-divisor heuristic before emitting runtime checks.
    // Dynamic divisors are checked by prepare_range.
    if matches!(const_index(ctx, d), Some(0..=2)) {
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

/// Reject cheap cases before doing any division or overflow preparation. The
/// placeholder bounds are never used by the quotient loop. Range indices are
/// scalar, so the eligibility vector has just one lane.
#[cube]
fn prepare_range<I: Int, N: Size>(
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    base: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
) -> QuotientRangeValues<I, N> {
    let eligible = d
        .greater_equal(&Vector::new(I::new(3)))
        .vec_and(s.greater_equal(&Vector::new(I::new(1))))
        .vec_and(lo.less_than(&hi));
    let mut start = Vector::new(I::new(0));
    let mut last = Vector::new(I::new(0));
    let mut end = Vector::new(I::new(0));
    let mut applicable = Vector::new(false);
    if eligible.extract(0usize) {
        let product_high = high_product::<I, N>(hi, s);
        let range = quotient_bounds::<I, N>(lo, hi, base, s, d, product_high);
        start = range.start;
        last = range.last;
        end = range.end;
        applicable = range.applicable;
    }
    QuotientRangeValues::<I, N> {
        start,
        last,
        end,
        applicable,
    }
}

#[cube]
fn high_product<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    intrinsic!(|scope| {
        polyfills::expand_himul_sim(scope, lhs.value(scope), rhs.value(scope)).into()
    })
}

/// Bracket quotients using both endpoints. Keep the arithmetic total even if a
/// backend speculates it: saturate subtraction and clamp the divisor.
#[cube]
fn quotient_bounds<I: Int, N: Size>(
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    base: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
    product_high: Vector<I, N>,
) -> QuotientRangeValues<I, N> {
    let s_safe = s.max(Vector::new(I::new(1)));
    // Eligibility guarantees d >= 3. Retain the clamp so speculative evaluation
    // is also safe, including q_last + 1 when the source divisor is 0 or 1.
    let d_safe = d.max(Vector::new(I::new(3)));
    let widest = base - (lo * s_safe).min(base);
    let upper_product = hi * s_safe;
    let narrowest = base - upper_product.min(base);
    // Exclude q for i == hi: base - q*d must be strictly less than hi*s.
    let past_end = Vector::<I, N>::cast_from(base.greater_equal(&upper_product));
    let q_start = narrowest / d_safe + past_end;
    let q_last = widest / d_safe;
    let q_end = q_last + Vector::new(I::new(1));
    // The high half of hi*s is zero exactly when the endpoint fits. This
    // avoids a dynamic integer division in the loop-entry safety predicate.
    let no_overflow = product_high.equal(&Vector::new(I::new(0)));
    // Heuristic: skip at least as many iterations as we keep. This is not a
    // target-specific cost model; benchmarks must still establish the benefit.
    let applicable =
        no_overflow.vec_and((q_end - q_start).less_equal(&((hi - lo) / Vector::new(I::new(2)))));
    QuotientRangeValues::<I, N> {
        start: q_start,
        last: q_last,
        end: q_end,
        applicable,
    }
}

/// Descending quotients restore source order. Bounds guarantee `tap*d <= base`;
/// exactness and range checks validate i. The other results replace arithmetic
/// already present in the body: `tap*d = base-i*s` and `scaled = i*s`.
#[cube]
fn recover_index<I: Int, N: Size>(
    q: Vector<I, N>,
    lo: Vector<I, N>,
    hi: Vector<I, N>,
    base: Vector<I, N>,
    s: Vector<I, N>,
    d: Vector<I, N>,
    q_start: Vector<I, N>,
    q_last: Vector<I, N>,
    #[comptime] unit_stride: bool,
) -> RecoveredIndex<I, N> {
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
    RecoveredIndex::<I, N> {
        index: recovered,
        quotient: tap,
        numerator,
        scaled,
        keep,
    }
}

impl Guarded {
    /// Only move our own total preparation across immediately enclosing range
    /// loops whose bodies define none of its inputs. Valid SSA then guarantees
    /// those inputs also dominate the insertion point before the outer loop.
    /// In particular, do not cross an if or move any original loads/computation.
    fn preparation_point(&self, ctx: &Context) -> Ptr<Operation> {
        let inputs = [
            self.loop_op.start(ctx),
            self.loop_op.end(ctx),
            self.division.base,
            self.division.scale,
            self.division.divisor,
        ];
        let mut point = self.loop_op.get_operation();
        while let Some(parent) = point.deref(ctx).get_parent_op(ctx) {
            let Some(outer) = parent.as_op::<RangeLoopOp>(ctx) else {
                break;
            };
            if !inputs
                .iter()
                .all(|&v| invariant(ctx, v, outer.loop_region(ctx)))
            {
                break;
            }
            point = parent;
        }
        point
    }

    fn prepare(&self, scope: &Scope) -> QuotientRangeValuesExpand<Idx, Lane> {
        let lo = self.loop_op.start(scope.ctx());
        let div = self.division;
        scope.register_value_type::<Idx, Lane>(lo);
        let range = prepare_range::expand::<Idx, Lane>(
            scope,
            lo.into(),
            self.loop_op.end(scope.ctx()).into(),
            div.base.into(),
            div.scale.into(),
            div.divisor.into(),
        );
        // The conditional polyfill returns mutable locals. Materialize their
        // values here so all consumers, including the loop bounds, use values
        // rather than local pointers, and the reads are outside the outer loop.
        QuotientRangeValuesExpand {
            start: range.start.read_value(scope).into(),
            last: range.last.read_value(scope).into(),
            end: range.end.read_value(scope).into(),
            applicable: range.applicable.read_value(scope).into(),
        }
    }

    fn recover(
        &self,
        scope: &Scope,
        q: Value,
        range: &QuotientRangeValuesExpand<Idx, Lane>,
    ) -> RecoveredIndexExpand<Idx, Lane> {
        let div = self.division;
        recover_index::expand::<Idx, Lane>(
            scope,
            q.into(),
            self.loop_op.start(scope.ctx()).into(),
            self.loop_op.end(scope.ctx()).into(),
            div.base.into(),
            div.scale.into(),
            div.divisor.into(),
            range.start,
            range.last,
            const_index(scope.ctx(), div.scale) == Some(1),
        )
    }

    /// Clone the body unchanged, then substitute proven values in the clone.
    /// SCCP/DCE remove the redundant guards and definitions afterwards.
    fn clone_body(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        dst: Ptr<BasicBlock>,
        tap: &RecoveredIndexExpand<Idx, Lane>,
    ) {
        let (index, scaled, numerator, quotient) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            (
                tap.index.value(&scope),
                tap.scaled.value(&scope),
                tap.numerator.value(&scope),
                tap.quotient.value(&scope),
            )
        };
        let truth = ConstantOp::new(ctx, Box::new(BoolAttr::new(true)));
        rewriter.append_op(ctx, &truth);
        let truth = truth.get_result(ctx);
        let mut mapper = IrMapping::new();
        mapper.map_value(self.loop_op.iter_var(ctx), index);
        let ops = self
            .loop_op
            .loop_body(ctx)
            .deref(ctx)
            .iter(ctx)
            .collect::<Vec<_>>();
        for old in ops {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(dst));
            let cloned = clone_operation(old, ctx, rewriter, &mut mapper);
            rewriter.append_operation(ctx, cloned);
        }
        let div = self.division;
        let mut replacements = vec![
            (self.product, scaled),
            (div.numerator, numerator),
            (div.remainder, mapper.lookup_value_or_default(div.zero)),
            (self.outer.condition(ctx), truth),
            (self.gate.condition(ctx), truth),
        ];
        for usage in div.numerator.uses(ctx) {
            if let Some(op) = usage.user_op().as_op::<UDivOp>(ctx)
                && op.lhs(ctx) == div.numerator
                && op.rhs(ctx) == div.divisor
            {
                replacements.push((op.get_result(ctx), quotient));
            }
        }
        for (original, replacement) in replacements {
            if let Some(cloned) = mapper.lookup_value(original) {
                cloned.replace_all_uses_with(ctx, &replacement);
            }
        }
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
        mark_unswitched(ctx, op);

        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(
            guard.preparation_point(ctx),
        ));
        let range = guard.prepare(&Scope::from_context_and_inserter(ctx, rewriter));
        let (start, end, applicable) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            (
                range.start.value(&scope),
                range.end.value(&scope),
                range.applicable.value(&scope),
            )
        };
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        let unswitch = IfOp::new(ctx, applicable);
        rewriter.append_op(ctx, &unswitch);
        let (then_block, else_block) = (unswitch.then_block(ctx), unswitch.else_block(ctx));

        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(then_block));
        let q_loop = RangeLoopOp::new(ctx, start, end, guard.loop_op.step(ctx));
        mark_unswitched(ctx, q_loop.get_operation());
        rewriter.append_op(ctx, &q_loop);
        let q = q_loop.iter_var(ctx);
        let q_body = q_loop.loop_body(ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(q_body));
        let tap = guard.recover(&Scope::from_context_and_inserter(ctx, rewriter), q, &range);
        let keep = tap
            .keep
            .value(&Scope::from_context_and_inserter(ctx, rewriter));
        let body_gate = IfOp::new(ctx, keep);
        rewriter.append_op(ctx, &body_gate);

        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(body_gate.get_operation()));
        guard.clone_body(ctx, rewriter, body_gate.then_block(ctx), &tap);
        for block in [q_body, body_gate.else_block(ctx), then_block, else_block] {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(block));
            let terminator = YieldOp::new(ctx);
            rewriter.append_op(ctx, &terminator);
        }
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(else_block));
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockStart(else_block));
        Ok(())
    }
}
