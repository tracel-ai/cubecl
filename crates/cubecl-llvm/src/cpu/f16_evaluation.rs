//! How wide f16 arithmetic is evaluated in.
//!
//! x86 has no half arithmetic short of AVX512-FP16, so the backend wraps every f16 operation in
//! a convert pair. Running a chain in f32 and rounding once at its end pays that pair once
//! instead of once per operation, and is what `gcc` and `clang` do with `_Float16` by default;
//! per-operation rounding is what they emit only under `-fexcess-precision=16`.
//!
//! Extending that across a loop-carried accumulator goes further than any C compiler does, and
//! doubles the bytes a long-lived value occupies in vector registers, so it is asked for
//! separately.

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::attributes::IndexAttr;
use cubecl_core::ir::dialect::cmp::{FMaxOp, FMinOp};
use cubecl_core::ir::dialect::general::CastOp;
use cubecl_core::ir::dialect::math::{
    FAddOp, FDivOp, FMulOp, FNegOp, FRemOp, FSubOp, FmaOp, RecipOp, RsqrtOp, SqrtOp,
};
use cubecl_core::ir::dialect::memory::{DeclareVariableOp, LoadOp, StoreOp};
use cubecl_core::ir::interfaces::{MaterializableOp, TypedExt};
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::try_cast_op;
use cubecl_core::ir::types::scalar::{Float16Type, Float32Type};
use cubecl_core::ir::types::{PointerType, VectorType};
use cubecl_environment::collections::HashMap;
use pliron::builtin::ops::FuncOp;
use pliron::graph::walkers::{WALKCONFIG_PREORDER_FORWARD, uninterruptible::mutable::walk_op};

/// How far an f32 intermediate is allowed to travel before it is rounded back to f16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum F16Evaluation {
    /// Round after every operation. What the hardware would do if it had f16 arithmetic, and
    /// what a GPU does, at a convert pair per operation.
    PerOperation,
    /// Round at the end of a chain of arithmetic. The default, and what a C compiler does.
    #[default]
    Chain,
    /// Round at the end of a chain, and hold a loop-carried accumulator in f32 across the back
    /// edge as well. Costs vector registers, so a wide kernel may want a narrower line.
    Accumulators,
}

/// Rewrites f16 arithmetic to f32 arithmetic between a widening and a narrowing convert, reusing
/// the f32 value where one rewritten operation feeds another.
pub struct EvaluateF16Pass {
    /// Whether a private variable of f16 is retyped, which is what carries a chain across a loop.
    pub accumulators: bool,
}

#[pass_name]
impl Pass for EvaluateF16Pass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        if op.as_op::<FuncOp>(ctx).is_none() {
            return Ok(res);
        }

        let mut found = Found {
            accumulators: self.accumulators,
            ..Default::default()
        };
        walk_op(
            ctx,
            &mut found,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, found, node| {
                if let IRNode::Operation(op) = node {
                    if is_f16_arithmetic(ctx, op) {
                        found.arithmetic.push(op);
                    } else if found.accumulators
                        && let Some(variable) = op.as_op::<DeclareVariableOp>(ctx)
                        && is_f16_local(ctx, &variable)
                    {
                        found.variables.push(variable);
                    }
                }
            },
        );

        if found.arithmetic.is_empty() {
            return Ok(res);
        }

        // A preorder walk visits definitions before uses, so an operand that a previous
        // iteration narrowed is already in `widened` and its f32 source is reused directly.
        let mut widened = HashMap::default();
        let mut rewriter = PassRewriter::default();
        for target in found.arithmetic {
            promote(ctx, &mut widened, &mut rewriter, target);
        }
        // After the arithmetic, so that a variable is judged on the converts the rewrite
        // actually left around it.
        for variable in found.variables {
            promote_variable(ctx, &mut widened, &mut rewriter, variable);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

#[derive(Default)]
struct Found {
    accumulators: bool,
    arithmetic: Vec<Ptr<Operation>>,
    variables: Vec<DeclareVariableOp>,
}

fn is_f16_arithmetic(ctx: &Context, op: Ptr<Operation>) -> bool {
    let promotable = op.is_op::<FAddOp>(ctx)
        || op.is_op::<FSubOp>(ctx)
        || op.is_op::<FMulOp>(ctx)
        || op.is_op::<FDivOp>(ctx)
        || op.is_op::<FRemOp>(ctx)
        || op.is_op::<FNegOp>(ctx)
        || op.is_op::<FmaOp>(ctx)
        || op.is_op::<FMinOp>(ctx)
        || op.is_op::<FMaxOp>(ctx)
        || op.is_op::<SqrtOp>(ctx)
        || op.is_op::<RsqrtOp>(ctx)
        || op.is_op::<RecipOp>(ctx);

    promotable && op.deref(ctx).get_num_results() == 1 && is_f16(ctx, op.deref(ctx).get_result(0))
}

fn is_f16(ctx: &Context, value: impl Typed) -> bool {
    scalar_is::<Float16Type>(ctx, value)
}

/// Guarded rather than a plain `scalar_ty`, which panics on the types that carry no element at
/// all: a barrier, a matrix, a tensor map.
fn scalar_is<T: pliron::r#type::Type>(ctx: &Context, value: impl Typed) -> bool {
    value
        .try_get_scalar_elem_ty(ctx)
        .is_some_and(|scalar| scalar.deref(ctx).is::<T>())
}

fn promote(
    ctx: &mut Context,
    widened: &mut HashMap<Value, Value>,
    rewriter: &mut PassRewriter,
    op: Ptr<Operation>,
) {
    let result = op.deref(ctx).get_result(0);
    let narrow_ty = result.get_type(ctx);
    let wide_ty = widen_ty(ctx, narrow_ty);

    let operands: Vec<Value> = op
        .operands(ctx)
        .into_iter()
        .map(|operand| widen(ctx, widened, operand, op))
        .collect();

    let attributes = op.deref(ctx).attributes.clone();
    let dyn_op = op.dyn_op(ctx);
    let rematerialize = try_cast_op!(dyn_op, ctx, dyn MaterializableOp);
    let wide_op = rematerialize.materialize(ctx, vec![wide_ty], operands, attributes);
    wide_op.insert_before(ctx, op);

    let wide_result = wide_op.deref(ctx).get_result(0);
    let narrowed = CastOp::new(ctx, narrow_ty, wide_result);
    narrowed.get_operation().insert_before(ctx, op);

    let narrowed = narrowed.get_result(ctx);
    result.replace_all_uses_with(ctx, &narrowed);
    widened.insert(narrowed, wide_result);
    rewriter.erase_operation(ctx, op);
}

/// The f32 form of `value`, converting it where it is not already one this pass produced.
///
/// The convert goes next to the definition rather than next to the use, so that it dominates
/// every later use and one value is never converted twice.
fn widen(
    ctx: &mut Context,
    widened: &mut HashMap<Value, Value>,
    value: Value,
    user: Ptr<Operation>,
) -> Value {
    if let Some(wide) = widened.get(&value) {
        return *wide;
    }

    let wide_ty = widen_ty(ctx, value.get_type(ctx));
    let cast = CastOp::new(ctx, wide_ty, value);
    match value.defining_op() {
        Some(definition) => cast.get_operation().insert_after(ctx, definition),
        None => match value.get_defining_block(ctx) {
            Some(block) => cast.get_operation().insert_at_front(block, ctx),
            None => cast.get_operation().insert_before(ctx, user),
        },
    }

    let wide = cast.get_result(ctx);
    widened.insert(value, wide);
    wide
}

/// A private variable of f16, which is how a loop carries an accumulator before `mem2reg` turns
/// it into a value. An initializer would have to be rewritten with it, and an accumulator gets
/// its starting value from a store instead.
fn is_f16_local(ctx: &Context, variable: &DeclareVariableOp) -> bool {
    variable.addr_space(ctx).0 == AddressSpace::Local
        && variable.initializer(ctx).is_none()
        && is_f16(ctx, variable.value_ty(ctx).get_type(ctx))
}

/// Holds the variable in f32 so that a loop reading and writing it every iteration stops
/// converting on both sides.
///
/// A widening convert on a load and a narrowing one on a store both disappear; every other use
/// of a load and every other stored value gains one. Converts under a loop the variable is
/// declared outside of are counted on their own and decide first, because one paid every
/// iteration outweighs any fixed number at the boundary, whatever the two counts are.
fn promote_variable(
    ctx: &mut Context,
    widened: &mut HashMap<Value, Value>,
    rewriter: &mut PassRewriter,
    variable: DeclareVariableOp,
) {
    let pointer = variable.get_result(ctx);
    let mut loads = Vec::new();
    let mut stores = Vec::new();

    for r#use in pointer.uses(ctx) {
        let user = r#use.user_op();
        if let Some(load) = user.as_op::<LoadOp>(ctx) {
            loads.push(load);
        } else if let Some(store) = user.as_op::<StoreOp>(ctx)
            && store.ptr(ctx) == pointer
        {
            stores.push(store);
        } else {
            return;
        }
    }

    let declared_at = nesting(ctx, variable.get_operation());
    let mut per_iteration = Converts::default();
    let mut once = Converts::default();
    let mut ledger = |ctx: &Context, op: Ptr<Operation>, removed: bool| {
        match nesting(ctx, op) > declared_at {
            true => &mut per_iteration,
            false => &mut once,
        }
        .record(removed);
    };

    for load in &loads {
        for r#use in load.get_result(ctx).uses(ctx) {
            let user = r#use.user_op();
            ledger(ctx, user, is_widening(ctx, user));
        }
    }
    for store in &stores {
        let removed = is_narrowed(ctx, store.value(ctx));
        ledger(ctx, store.get_operation(), removed);
    }

    let worth_it = per_iteration.verdict().or_else(|| once.verdict());
    if worth_it != Some(true) {
        return;
    }

    let narrow_ty = variable.value_ty(ctx).get_type(ctx);
    let wide_ty = widen_ty(ctx, narrow_ty);
    let address_space = variable.addr_space(ctx).0;
    variable.set_value_ty(ctx, wide_ty);
    variable.set_alignment(ctx, IndexAttr(wide_ty.align(ctx)));
    pointer.set_type(ctx, PointerType::get(ctx, wide_ty, address_space).into());

    for load in loads {
        let loaded = load.get_result(ctx);
        // Both lists come first: the retype makes a widening convert stop looking like one, and
        // erasing one hands its users to the load, where they then look like f16 consumers.
        let (widening, narrowing): (Vec<_>, Vec<_>) = loaded
            .uses(ctx)
            .into_iter()
            .partition(|r#use| is_widening(ctx, r#use.user_op()));

        widened.remove(&loaded);
        loaded.set_type(ctx, wide_ty);

        for r#use in widening {
            let cast = r#use.user_op();
            let wide = cast.deref(ctx).get_result(0);
            widened.remove(&wide);
            wide.replace_all_uses_with(ctx, &loaded);
            rewriter.erase_operation(ctx, cast);
        }

        if !narrowing.is_empty() {
            let cast = CastOp::new(ctx, narrow_ty, loaded);
            cast.get_operation().insert_after(ctx, load.get_operation());
            let narrowed = cast.get_result(ctx);
            for r#use in narrowing {
                loaded.replace_use_with(ctx, r#use, &narrowed);
            }
            widened.insert(narrowed, loaded);
        }
    }

    for store in stores {
        let stored = store.value(ctx);
        let wide = widen(ctx, widened, stored, store.get_operation());
        let op = store.get_operation();
        op.operand(ctx, 1)
            .replace_use_with(ctx, op.operand_as_use(ctx, 1), &wide);
    }
}

/// Converts a promotion would remove against ones it would add.
#[derive(Default)]
struct Converts {
    removed: usize,
    added: usize,
}

impl Converts {
    fn record(&mut self, removed: bool) {
        match removed {
            true => self.removed += 1,
            false => self.added += 1,
        }
    }

    /// `None` where the two cancel, which leaves the decision to the next ledger.
    fn verdict(&self) -> Option<bool> {
        (self.removed != self.added).then_some(self.removed > self.added)
    }
}

fn nesting(ctx: &Context, op: Ptr<Operation>) -> usize {
    let mut depth = 0;
    let mut current = op;
    while let Some(parent) = current.deref(ctx).get_parent_op(ctx) {
        depth += 1;
        current = parent;
    }
    depth
}

fn is_widening(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.is_op::<CastOp>(ctx)
        && is_f16(ctx, op.operand(ctx, 0))
        && scalar_is::<Float32Type>(ctx, op.deref(ctx).get_result(0))
}

fn is_narrowing(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.is_op::<CastOp>(ctx)
        && scalar_is::<Float32Type>(ctx, op.operand(ctx, 0))
        && is_f16(ctx, op.deref(ctx).get_result(0))
}

/// Whether widening `value` again would cost nothing, because it is the f16 side of a convert.
fn is_narrowed(ctx: &Context, value: Value) -> bool {
    value.defining_op().is_some_and(|op| is_narrowing(ctx, op))
}

fn widen_ty(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let f32_ty = Float32Type::get(ctx).to_handle();
    match ty.deref(ctx).downcast_ref::<VectorType>() {
        Some(vector) => {
            let vectorization = vector.vectorization;
            VectorType::get(ctx, f32_ty, vectorization).to_handle()
        }
        None => f32_ty,
    }
}
