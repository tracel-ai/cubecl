//! How wide f16 arithmetic is evaluated in.
//!
//! Without AVX512-FP16, x86 has no f16 arithmetic at all, so the backend wraps every f16
//! operation in a convert pair. Running a chain in f32 and rounding once at its end pays that
//! pair once instead of once per operation. `gcc` and `clang` do the same with `_Float16` within
//! an expression but round at every assignment, where a chain here follows SSA values: it runs
//! through an immutable `let` and across operations the fusion layer joined.
//!
//! Holding a private variable in f32 as well goes further than any C compiler does, and doubles
//! the bytes a long-lived value occupies in vector registers, so it is asked for separately.

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::attributes::IndexAttr;
use cubecl_core::ir::dialect::branch::{RangeLoopOp, WhileOp};
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
    /// Round where a value is stored, a `let mut` included, or read by anything but arithmetic.
    /// The default.
    #[default]
    Chain,
    /// Also hold a private f16 variable in f32 where that removes more converts than it adds, so
    /// a running total read often enough inside its loop stays in f16. Costs vector registers, so
    /// a wide kernel may want a narrower line.
    Accumulators,
}

impl F16Evaluation {
    /// Every mode, so that a caller offering the choice cannot miss one.
    pub const ALL: [Self; 3] = [Self::PerOperation, Self::Chain, Self::Accumulators];

    fn name(self) -> &'static str {
        match self {
            Self::PerOperation => "per-operation",
            Self::Chain => "chain",
            Self::Accumulators => "accumulators",
        }
    }
}

impl core::fmt::Display for F16Evaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

impl core::str::FromStr for F16Evaluation {
    type Err = UnknownF16Evaluation;

    fn from_str(name: &str) -> core::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.name() == name)
            .ok_or_else(|| UnknownF16Evaluation(name.to_string()))
    }
}

/// A name no [`F16Evaluation`] spells.
#[derive(Debug)]
pub struct UnknownF16Evaluation(String);

impl core::fmt::Display for UnknownF16Evaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let known = F16Evaluation::ALL.map(F16Evaluation::name).join(", ");
        write!(f, "{} is not one of {known}", self.0)
    }
}

impl core::error::Error for UnknownF16Evaluation {}

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
        && is_widenable_f16(ctx, variable.value_ty(ctx).get_type(ctx))
}

/// f16 or a vector of it, which is all [`widen_ty`] knows: an atomic, an array or a matrix of
/// f16 answers to [`is_f16`] just as readily and would come back a bare `f32`.
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

/// The accesses to `variable`, or `None` where its pointer reaches anything but a load or a store
/// of its own, since retyping it would change what that op reads.
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

/// The candidates a copy joins, as groups of index into `candidates`. A copy between two f16
/// locals stops converting only when both sides are held, so they are weighed and held together.
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

    // By candidate order rather than by root, so the promotions run in the same order every time.
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

/// Holds a group of variables in f32 so that a loop reading and writing them every iteration stops
/// converting on both sides.
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

/// Converts under a loop the group is declared outside of decide first, because one paid every
/// iteration outweighs any fixed number at the boundary, whatever the two counts are.
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
    // The outermost declaration anchors the group, so one ledger covers every member of it.
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
            for r#use in load.get_result(ctx).uses(ctx) {
                let user = r#use.user_op();
                if stores_into(ctx, user, &inside) {
                    continue;
                }
                ledger(ctx, user, is_widening(ctx, user));
            }
        }
        for store in &candidate.stores {
            let stored = store.value(ctx);
            if loaded_from(ctx, stored).is_some_and(|source| inside.contains(&source)) {
                continue;
            }
            // Only a convert this pass made is bypassed; one the kernel wrote is kept and widened.
            ledger(ctx, store.get_operation(), widened.contains_key(&stored));
        }
    }

    per_iteration.verdict().or_else(|| once.verdict()) == Some(true)
}

/// The variable `value` was loaded from, where it is a load at all.
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

/// Loops only. An `if` nests a region of its own, and counting one would let a convert paid
/// once decide the ledger that exists for converts paid every iteration.
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
