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
        general::BoolNotOp,
        math::{IMulOp, ISubOp, UDivOp, URemOp},
    },
    ident,
    interfaces::TypedExt,
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
struct QuotientRangeValues {
    start: usize,
    last: usize,
    end: usize,
    applicable: bool,
}

#[derive(CubeType)]
#[allow(dead_code)] // Fields are read through the generated expansion type.
struct RecoveredIndex {
    index: usize,
    quotient: usize,
    numerator: usize,
    scaled: usize,
    keep: bool,
}

#[derive(CubeType)]
#[allow(dead_code)]
struct SmallDivision {
    quotient: usize,
    remainder: usize,
}

/// Specialize the common small divisors without preparing a quotient range.
#[cube]
fn small_division(numerator: usize, divisor: usize, #[comptime] by_three: bool) -> SmallDivision {
    if by_three {
        SmallDivision {
            quotient: numerator / 3,
            remainder: numerator % 3,
        }
    } else {
        let bits = divisor - 1;
        SmallDivision {
            quotient: numerator >> bits,
            remainder: numerator & bits,
        }
    }
}

#[cube]
fn is_small_divisor(divisor: usize) -> bool {
    divisor == 1 || divisor == 2
}

/// Constant division is cheaper than range preparation for a single tiny
/// window. Repeated or larger windows still benefit from quotient traversal.
#[cube]
fn use_constant_three(
    lo: usize,
    hi: usize,
    scale: usize,
    divisor: usize,
    outer_start: usize,
    outer_end: usize,
    outer_step: usize,
) -> bool {
    let span = hi - lo.min(hi);
    let outer_span = outer_end - outer_start.min(outer_end);
    let reused = outer_step > 0 && outer_span > outer_step;
    divisor == 3 && (scale != 1 || (span <= 3 && !reused))
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
    } else {
        let cmp = defining.as_op::<ULessThanOrEqualOp>(ctx)?;
        (cmp.lhs(ctx), cmp.rhs(ctx))
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
    // Reject known ineligible strides/divisors before cloning the loop.
    // Dynamic values are checked by prepare_range.
    match (const_index(ctx, s), const_index(ctx, d)) {
        (Some(0), _) | (_, Some(0..=2)) => return None,
        (Some(s), Some(d)) if s > d / 2 => return None,
        _ => {}
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

/// Prepare scalar `cube.index` bounds; `usize` follows the backend's index width.
/// Cheap rejections bypass bound and overflow preparation entirely.
#[cube]
fn prepare_range(
    lo: usize,
    hi: usize,
    base: usize,
    scale: usize,
    divisor: usize,
    small_three: bool,
) -> QuotientRangeValues {
    let mut start = 0usize;
    let mut last = 0usize;
    let mut end = 0usize;
    let mut applicable = false;
    // The common small-divisor fallback needs only this check. Keep the
    // remaining eligibility arithmetic inside the branch.
    if divisor >= 3 && !small_three {
        // Require a 2x reduction in density; unreachable iterations are cheap.
        let eligible = scale >= 1 && scale <= divisor / 2 && lo < hi;
        if eligible {
            // Keep divisions and q_last + 1 safe even under speculative evaluation.
            let divisor_safe = divisor.max(3);
            let widest = base - (lo * scale).min(base);
            let upper_product = hi * scale;
            let narrowest = base - upper_product.min(base);
            // Exclude q for i == hi: base - q*divisor must be less than hi*scale.
            let past_end = (base >= upper_product) as usize;
            let q_start = narrowest / divisor_safe + past_end;
            let q_last = widest / divisor_safe;
            let q_end = q_last + 1;
            // A zero high half proves hi*scale fits without a dynamic division.
            let no_overflow = high_product(hi, scale) == 0;
            start = q_start;
            last = q_last;
            end = q_end;
            // Keep at most half as many candidates as source iterations.
            applicable = no_overflow && q_end - q_start <= (hi - lo) / 2;
        }
    }
    QuotientRangeValues {
        start,
        last,
        end,
        applicable,
    }
}

#[cube]
fn high_product(lhs: usize, rhs: usize) -> usize {
    intrinsic!(|scope| {
        polyfills::expand_himul_sim(scope, lhs.value(scope), rhs.value(scope)).into()
    })
}

/// Descending quotients preserve source order. Exactness and range checks
/// validate the recovered index; the other values replace body arithmetic.
#[cube]
fn recover_index(
    q: usize,
    lo: usize,
    hi: usize,
    base: usize,
    scale: usize,
    divisor: usize,
    q_start: usize,
    q_last: usize,
    #[comptime] unit_stride: bool,
) -> RecoveredIndex {
    let tap = q_last - (q - q_start);
    let numerator = tap * divisor;
    let scaled = base - numerator;
    let recovered = if unit_stride { scaled } else { scaled / scale };
    let exact = if unit_stride {
        true.runtime()
    } else {
        scaled.is_multiple_of(scale)
    };
    let keep = exact && (recovered >= lo && recovered < hi);
    RecoveredIndex {
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
    /// Do not cross an if or move any original loads/computation. A loop nest
    /// with only pure siblings can also select its version outside the nest;
    /// stop at other regions to avoid multiplying unrelated loop bodies.
    fn insertion_points(&self, ctx: &Context) -> (Ptr<Operation>, Ptr<Operation>) {
        let inputs = [
            self.loop_op.start(ctx),
            self.loop_op.end(ctx),
            self.division.base,
            self.division.scale,
            self.division.divisor,
        ];
        let mut point = self.loop_op.get_operation();
        let mut dispatch = point;
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
            if dispatch == point
                && outer.loop_body(ctx).deref(ctx).iter(ctx).all(|op| {
                    op == point
                        || op.is_op::<YieldOp>(ctx)
                        || (op.regions(ctx).is_empty() && !has_effects(ctx, op))
                })
            {
                dispatch = parent;
            }
            point = parent;
        }
        (point, dispatch)
    }

    fn small_three(&self, scope: &Scope, preparation: Ptr<Operation>) -> NativeExpand<bool> {
        let bounds = if preparation != self.loop_op.get_operation()
            && let Some(outer) = preparation.as_op::<RangeLoopOp>(scope.ctx())
            && outer.start(scope.ctx()).is_index(scope.ctx())
            && outer.start(scope.ctx()).vector_size(scope.ctx()) == 1
        {
            [
                outer.start(scope.ctx()),
                outer.end(scope.ctx()),
                outer.step(scope.ctx()),
            ]
            .map(Into::into)
        } else {
            [0usize, 1, 1].map(|value| value.into_expand(scope))
        };
        use_constant_three::expand(
            scope,
            self.loop_op.start(scope.ctx()).into(),
            self.loop_op.end(scope.ctx()).into(),
            self.division.scale.into(),
            self.division.divisor.into(),
            bounds[0],
            bounds[1],
            bounds[2],
        )
        .read_value(scope)
        .into()
    }

    fn prepare(&self, scope: &Scope, small_three: NativeExpand<bool>) -> QuotientRangeValuesExpand {
        let lo = self.loop_op.start(scope.ctx());
        let div = self.division;
        let range = prepare_range::expand(
            scope,
            lo.into(),
            self.loop_op.end(scope.ctx()).into(),
            div.base.into(),
            div.scale.into(),
            div.divisor.into(),
            small_three,
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
        range: &QuotientRangeValuesExpand,
    ) -> RecoveredIndexExpand {
        let div = self.division;
        recover_index::expand(
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

    /// Select unit-scale recovery once per nest, without a branch per candidate.
    fn specialize_unit_scale(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        nest: Ptr<Operation>,
    ) {
        if const_index(ctx, self.division.scale).is_some() {
            return;
        }
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(nest));
        let one = ConstantOp::new(ctx, Box::new(IndexAttr::new(1)));
        rewriter.append_op(ctx, &one);
        let one = one.get_result(ctx);
        let condition = IEqualOp::new(ctx, self.division.scale, one);
        rewriter.append_op(ctx, &condition);
        let choice = IfOp::new(ctx, condition.get_result(ctx));
        rewriter.append_op(ctx, &choice);
        let mut mapper = IrMapping::new();
        mapper.map_value(self.division.scale, one);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(choice.then_block(ctx)));
        let unit = clone_operation(nest, ctx, rewriter, &mut mapper);
        rewriter.append_operation(ctx, unit);
        rewriter.move_operation(
            ctx,
            nest,
            OpInsertionPoint::AtBlockStart(choice.else_block(ctx)),
        );
        for block in [choice.then_block(ctx), choice.else_block(ctx)] {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(block));
            let terminator = YieldOp::new(ctx);
            rewriter.append_op(ctx, &terminator);
        }
    }

    fn replace_quotients(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        mapper: &IrMapping,
        quotient: Value,
    ) {
        let div = self.division;
        let mut replacements = Vec::new();
        for usage in div.numerator.uses(ctx) {
            if let Some(op) = usage.user_op().as_op::<UDivOp>(ctx)
                && op.lhs(ctx) == div.numerator
                && op.rhs(ctx) == div.divisor
                && let Some(cloned) = mapper.lookup_value(op.get_result(ctx))
            {
                replacements.push(cloned);
            }
        }
        for cloned in replacements {
            cloned.replace_all_uses_with(ctx, &quotient);
            rewriter.erase_operation(ctx, cloned.defining_op().unwrap());
        }
    }

    /// Keep the original iteration order and guards, specializing only the
    /// division/remainder pair. No index recovery is needed on this path.
    fn clone_small_divisor(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        source: Ptr<Operation>,
        by_three: bool,
    ) {
        let mut mapper = IrMapping::new();
        let cloned = clone_operation(source, ctx, rewriter, &mut mapper);
        rewriter.append_operation(ctx, cloned);
        let div = self.division;
        let numerator = mapper.lookup_value(div.numerator).unwrap();
        let remainder = mapper.lookup_value(div.remainder).unwrap();
        rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(
            numerator.defining_op().unwrap(),
        ));
        let (quotient, replacement) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            let values = small_division::expand(
                &scope,
                numerator.into(),
                mapper.lookup_value_or_default(div.divisor).into(),
                by_three,
            );
            (
                values.quotient.value(&scope),
                values.remainder.value(&scope),
            )
        };
        remainder.replace_all_uses_with(ctx, &replacement);
        rewriter.erase_operation(ctx, remainder.defining_op().unwrap());
        self.replace_quotients(ctx, rewriter, &mapper, quotient);
    }

    /// Clone the body unchanged, then substitute proven values in the clone.
    /// SCCP/DCE remove the redundant guards and definitions afterwards.
    fn clone_body(
        &self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        dst: Ptr<BasicBlock>,
        tap: &RecoveredIndexExpand,
    ) {
        let (index, scaled, numerator, quotient) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            (
                tap.index.read_value(&scope),
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
        let replacements = [
            (self.product, scaled),
            (div.numerator, numerator),
            (div.remainder, mapper.lookup_value_or_default(div.zero)),
            (self.outer.condition(ctx), truth),
            (self.gate.condition(ctx), truth),
        ];
        for (original, replacement) in replacements {
            if let Some(cloned) = mapper.lookup_value(original) {
                cloned.replace_all_uses_with(ctx, &replacement);
                rewriter.erase_operation(ctx, cloned.defining_op().unwrap());
            }
        }
        self.replace_quotients(ctx, rewriter, &mapper, quotient);
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
        let (preparation, dispatch) = guard.insertion_points(ctx);
        let dynamic_divisor = const_index(ctx, guard.division.divisor).is_none();
        let (range, mut small_three) = if preparation != dispatch || !dynamic_divisor {
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(preparation));
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            let small = if dynamic_divisor {
                guard.small_three(&scope, preparation)
            } else {
                false.into_expand(&scope)
            };
            (Some(guard.prepare(&scope, small)), Some(small))
        } else {
            (None, None)
        };
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(dispatch));
        let mut quick_paths = Vec::new();
        if dynamic_divisor {
            for by_three in [false, true] {
                let small = {
                    let scope = Scope::from_context_and_inserter(ctx, rewriter);
                    if by_three {
                        small_three
                            .get_or_insert_with(|| guard.small_three(&scope, preparation))
                            .value(&scope)
                    } else {
                        is_small_divisor::expand(&scope, guard.division.divisor.into())
                            .value(&scope)
                    }
                };
                let quick = IfOp::new(ctx, small);
                rewriter.append_op(ctx, &quick);
                rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(quick.then_block(ctx)));
                guard.clone_small_divisor(ctx, rewriter, dispatch, by_three);
                rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(quick.then_block(ctx)));
                let yield_op = YieldOp::new(ctx);
                rewriter.append_op(ctx, &yield_op);
                rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(quick.else_block(ctx)));
                quick_paths.push(quick);
            }
        }
        let range = range.unwrap_or_else(|| {
            guard.prepare(
                &Scope::from_context_and_inserter(ctx, rewriter),
                small_three.unwrap(),
            )
        });
        let (start, end, applicable) = {
            let scope = Scope::from_context_and_inserter(ctx, rewriter);
            (
                range.start.value(&scope),
                range.end.value(&scope),
                range.applicable.value(&scope),
            )
        };
        let use_source = BoolNotOp::new(ctx, applicable);
        rewriter.append_op(ctx, &use_source);
        let unswitch = IfOp::new(ctx, use_source.get_result(ctx));
        rewriter.append_op(ctx, &unswitch);
        let (optimized, fallback) = (unswitch.else_block(ctx), unswitch.then_block(ctx));

        if dispatch != op {
            // Clone one fallback nest, then replace the original inner loop.
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(fallback));
            let original = clone_operation(dispatch, ctx, rewriter, &mut IrMapping::new());
            rewriter.append_operation(ctx, original);
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        } else {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(optimized));
        }
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
        for block in [q_body, body_gate.else_block(ctx), optimized, fallback] {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(block));
            let terminator = YieldOp::new(ctx);
            rewriter.append_op(ctx, &terminator);
        }
        let nest = if dispatch != op {
            rewriter.erase_operation(ctx, op);
            rewriter.move_operation(ctx, dispatch, OpInsertionPoint::AtBlockStart(optimized));
            dispatch
        } else {
            rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockStart(fallback));
            q_loop.get_operation()
        };
        guard.specialize_unit_scale(ctx, rewriter, nest);
        for quick in quick_paths {
            rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(quick.else_block(ctx)));
            let yield_op = YieldOp::new(ctx);
            rewriter.append_op(ctx, &yield_op);
        }
        Ok(())
    }
}
