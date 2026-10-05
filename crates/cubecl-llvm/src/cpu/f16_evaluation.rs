//! CPU f16 evaluation precision.

use crate::prelude::*;
use cubecl_core::ir::{
    dialect::{
        branch::{RangeLoopOp, WhileOp},
        cmp::{FMaxOp, FMinOp},
        general::CastOp,
        math::{FAddOp, FDivOp, FMulOp, FNegOp, FRemOp, FSubOp, FmaOp, RecipOp, RsqrtOp, SqrtOp},
        memory::{DeclareVariableOp, LoadOp, StoreOp},
    },
    try_cast_op,
    types::{
        PointerType, VectorType,
        scalar::{Float16Type, Float32Type},
    },
};
use cubecl_environment::collections::HashMap;
use pliron::graph::walkers::uninterruptible::mutable::walk_op;

/// F32 evaluation of f16 arithmetic.
pub struct EvaluateF16Pass {
    /// Allow f16 local variables to use f32 storage.
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

        let mut widened = HashMap::default();
        let mut rewriter = PassRewriter::default();
        for target in found.arithmetic {
            promote(ctx, &mut widened, &mut rewriter, target);
        }
        let candidates: Vec<Candidate> = found
            .variables
            .into_iter()
            .filter_map(|variable| accesses(ctx, variable))
            .collect();
        for group in copy_groups(ctx, &candidates) {
            promote_group(ctx, &mut widened, &mut rewriter, &candidates, &group);
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

fn is_f16_local(ctx: &Context, variable: &DeclareVariableOp) -> bool {
    variable.addr_space(ctx).0 == AddressSpace::Local
        && variable.initializer(ctx).is_none()
        && is_widenable_f16(ctx, variable.value_ty(ctx).get_type(ctx))
}

fn is_widenable_f16(ctx: &Context, ty: TypeHandle) -> bool {
    let elem = ty
        .deref(ctx)
        .downcast_ref::<VectorType>()
        .map(|vector| vector.inner)
        .unwrap_or(ty);

    elem.deref(ctx).is::<Float16Type>()
}

struct Candidate {
    variable: DeclareVariableOp,
    pointer: Value,
    loads: Vec<LoadOp>,
    stores: Vec<StoreOp>,
    declared_at: usize,
}

fn accesses(ctx: &Context, variable: DeclareVariableOp) -> Option<Candidate> {
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
            return None;
        }
    }

    Some(Candidate {
        variable,
        pointer,
        loads,
        stores,
        declared_at: loop_depth(ctx, variable.get_operation()),
    })
}

fn copy_groups(ctx: &Context, candidates: &[Candidate]) -> Vec<Vec<usize>> {
    let owner: HashMap<Value, usize> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.pointer, index))
        .collect();

    let mut parent: Vec<usize> = (0..candidates.len()).collect();
    for (index, candidate) in candidates.iter().enumerate() {
        for store in &candidate.stores {
            if let Some(source) = loaded_from(ctx, store.value(ctx))
                && let Some(&source) = owner.get(&source)
            {
                join(&mut parent, index, source);
            }
        }
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::default();
    for index in 0..candidates.len() {
        let root = root(&parent, index);
        match group_of.get(&root) {
            Some(&group) => groups[group].push(index),
            None => {
                group_of.insert(root, groups.len());
                groups.push(vec![index]);
            }
        }
    }
    groups
}

fn root(parent: &[usize], mut index: usize) -> usize {
    while parent[index] != index {
        index = parent[index];
    }
    index
}

fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    if a != b {
        parent[b] = a;
    }
}

fn promote_group(
    ctx: &mut Context,
    widened: &mut HashMap<Value, Value>,
    rewriter: &mut PassRewriter,
    candidates: &[Candidate],
    group: &[usize],
) {
    if !worth_holding(ctx, widened, candidates, group) {
        return;
    }

    for &index in group {
        hold_in_f32(ctx, widened, rewriter, &candidates[index]);
    }
}

fn worth_holding(
    ctx: &Context,
    widened: &HashMap<Value, Value>,
    candidates: &[Candidate],
    group: &[usize],
) -> bool {
    let inside: Vec<Value> = group
        .iter()
        .map(|&index| candidates[index].pointer)
        .collect();
    let declared_at = group
        .iter()
        .map(|&index| candidates[index].declared_at)
        .min()
        .unwrap_or(0);

    let mut per_iteration = Converts::default();
    let mut once = Converts::default();
    let mut ledger = |ctx: &Context, op: Ptr<Operation>, removed: bool| {
        match loop_depth(ctx, op) > declared_at {
            true => &mut per_iteration,
            false => &mut once,
        }
        .record(removed);
    };

    for &index in group {
        let candidate = &candidates[index];
        for load in &candidate.loads {
            let mut narrowed = false;
            for r#use in load.get_result(ctx).uses(ctx) {
                let user = r#use.user_op();
                if stores_into(ctx, user, &inside) {
                    continue;
                }
                match is_widening(ctx, user) {
                    true => ledger(ctx, user, true),
                    false => narrowed = true,
                }
            }
            if narrowed {
                ledger(ctx, load.get_operation(), false);
            }
        }
        for store in &candidate.stores {
            let stored = store.value(ctx);
            if loaded_from(ctx, stored).is_some_and(|source| inside.contains(&source)) {
                continue;
            }
            // Explicit kernel conversions must retain their rounding.
            ledger(ctx, store.get_operation(), widened.contains_key(&stored));
        }
    }

    per_iteration.verdict().or_else(|| once.verdict()) == Some(true)
}

fn loaded_from(ctx: &Context, value: Value) -> Option<Value> {
    let load = value.defining_op()?.as_op::<LoadOp>(ctx)?;
    Some(load.ptr(ctx))
}

fn stores_into(ctx: &Context, op: Ptr<Operation>, group: &[Value]) -> bool {
    op.as_op::<StoreOp>(ctx)
        .is_some_and(|store| group.contains(&store.ptr(ctx)))
}

fn hold_in_f32(
    ctx: &mut Context,
    widened: &mut HashMap<Value, Value>,
    rewriter: &mut PassRewriter,
    candidate: &Candidate,
) {
    let pointer = candidate.pointer;
    let narrow_ty = candidate.variable.value_ty(ctx).get_type(ctx);
    let wide_ty = widen_ty(ctx, narrow_ty);
    let address_space = candidate.variable.addr_space(ctx).0;
    candidate.variable.set_value_ty(ctx, wide_ty);
    candidate
        .variable
        .set_alignment(ctx, IndexAttr(wide_ty.align(ctx)));
    pointer.set_type(ctx, PointerType::get(ctx, wide_ty, address_space).into());

    for load in &candidate.loads {
        let loaded = load.get_result(ctx);
        let (widening, narrowing): (Vec<_>, Vec<_>) = loaded
            .uses(ctx)
            .into_iter()
            .partition(|r#use| is_widening(ctx, r#use.user_op()));

        widened.remove(&loaded);
        loaded.set_type(ctx, wide_ty);

        for r#use in widening {
            let cast = r#use.user_op();
            let wide = cast.deref(ctx).get_result(0);
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

    for store in &candidate.stores {
        let op = store.get_operation();
        let stored = store.value(ctx);
        let wide = widen(ctx, widened, stored, op);
        stored.replace_some_uses_with(ctx, |_, r#use| r#use.user_op() == op, &wide);
    }
}

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

    fn verdict(&self) -> Option<bool> {
        (self.removed != self.added).then_some(self.removed > self.added)
    }
}

fn loop_depth(ctx: &Context, op: Ptr<Operation>) -> usize {
    let mut depth = 0;
    let mut current = op;
    while let Some(parent) = current.deref(ctx).get_parent_op(ctx) {
        if parent.is_op::<RangeLoopOp>(ctx) || parent.is_op::<WhileOp>(ctx) {
            depth += 1;
        }
        current = parent;
    }
    depth
}

fn is_widening(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.is_op::<CastOp>(ctx)
        && is_f16(ctx, op.operand(ctx, 0))
        && scalar_is::<Float32Type>(ctx, op.deref(ctx).get_result(0))
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
