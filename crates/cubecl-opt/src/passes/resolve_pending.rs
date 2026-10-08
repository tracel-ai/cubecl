//! Resolves the waits on commit group tokens into waits on group counts.
//!
//! A wait on a token becomes a wait that lets `n` groups of the token's kind still run, where `n`
//! is the fewest groups committed after the token's group on any path from its commit to the
//! wait. Groups of one kind complete in commit order, so once at most `n` of the newest are left
//! running, the token's group is done on every path. Each token is tracked through the variables
//! that hold it; a wait on a token the analysis lost track of waits for every group.

use core::fmt;

use alloc::{string::ToString, vec::Vec};
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::{
        matrix::{WgmmaCommitGroupOp, WgmmaWaitGroupOp},
        memory::{DeclareVariableOp, LoadOp, StoreOp},
        pending::{CommitOp, ReadyOp, WaitOp},
        tma::{CommitGroupOp, WaitGroupReadOp},
    },
    prelude::*,
    rewrite::WALKCONFIG_ANY,
    types::{
        PointerType,
        pending::{AsyncGroup, GroupTokenType},
    },
};
use itertools::Itertools;
use pliron::{
    graph::walkers::uninterruptible::immutable::walk_op,
    irbuild::listener::DummyListener,
    printable::{self, Printable},
};

use crate::analyses::dataflow_solver::{
    ChangeResult, DataflowSolver, ProgramPoint, ReadRef, SolverConfig, WriteRef,
    dead_code::DeadCodeAnalysis,
    dense::{DenseForward, DenseForwardDataflowAnalysis, DenseLattice, LatticeValue},
    sccp::SparseConstantPropagationAnalysis,
};

/// Groups committed after a token's group. A token with nothing to wait for is infinitely far.
type Distance = u32;
const NOTHING_TO_WAIT: Distance = Distance::MAX;

/// For each token, and each variable holding one, the fewest groups of its kind committed since
/// its own on any path to this point.
#[derive(Default, PartialEq)]
pub struct GroupDistances(HashMap<Value, (AsyncGroup, Distance)>);

impl GroupDistances {
    fn get(&self, slot: Value, group: AsyncGroup) -> (AsyncGroup, Distance) {
        // A token the analysis lost track of may have just been committed.
        self.0.get(&slot).copied().unwrap_or((group, 0))
    }
}

impl Printable for GroupDistances {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self
            .0
            .iter()
            .map(|(slot, (_, distance))| (slot.disp(ctx).to_string(), *distance))
            .sorted()
            .map(|(slot, distance)| match distance {
                NOTHING_TO_WAIT => alloc::format!("{slot}: ready"),
                distance => alloc::format!("{slot}: {distance}"),
            })
            .join(", ");
        write!(f, "GroupDistances({{{entries}}})")
    }
}

impl LatticeValue for GroupDistances {
    /// Paths merge to the fewest commits either took.
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        let mut change = ChangeResult::Unchanged;
        for (slot, (group, distance)) in &rhs.0 {
            let entry = self.0.entry(*slot).or_insert_with(|| {
                change = ChangeResult::Changed;
                (*group, *distance)
            });
            if *distance < entry.1 {
                entry.1 = *distance;
                change = ChangeResult::Changed;
            }
        }
        change
    }
}

pub type GroupDistancesLattice = DenseLattice<GroupDistances>;

#[derive(Default)]
pub struct GroupDistancesAnalysis;

/// The kind of group `ty` is a token of, or of the token a variable of type `ty` holds.
fn token_group(ctx: &Context, ty: TypeHandle) -> Option<AsyncGroup> {
    let ty = ty.deref(ctx);
    if let Some(token) = ty.downcast_ref::<GroupTokenType>() {
        return Some(token.0);
    }
    let inner = ty.downcast_ref::<PointerType>()?.inner;
    inner
        .deref(ctx)
        .downcast_ref::<GroupTokenType>()
        .map(|token| token.0)
}

impl DenseForwardDataflowAnalysis for GroupDistancesAnalysis {
    type LatticeValue = GroupDistances;

    fn visit_operation(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        before: &ReadRef<GroupDistancesLattice>,
        after: &WriteRef<GroupDistancesLattice>,
    ) -> Result<()> {
        let mut distances = GroupDistances(before.deref().value().0.clone());
        let dyn_op = op.dyn_op(ctx);
        if let Some(commit) = dyn_op.downcast_ref::<CommitOp>() {
            let group = *commit.group(ctx);
            for (slot_group, distance) in distances.0.values_mut() {
                if *slot_group == group {
                    *distance = distance.saturating_add(1);
                }
            }
            distances.0.insert(commit.get_result(ctx), (group, 0));
        } else if let Some(ready) = dyn_op.downcast_ref::<ReadyOp>() {
            let group = *ready.group(ctx);
            distances
                .0
                .insert(ready.get_result(ctx), (group, NOTHING_TO_WAIT));
        } else if let Some(store) = dyn_op.downcast_ref::<StoreOp>() {
            let value = store.value(ctx);
            if let Some(group) = token_group(ctx, value.get_type(ctx)) {
                let distance = distances.get(value, group);
                distances.0.insert(store.ptr(ctx), distance);
            }
        } else if let Some(load) = dyn_op.downcast_ref::<LoadOp>() {
            let ptr = load.ptr(ctx);
            if let Some(group) = token_group(ctx, ptr.get_type(ctx)) {
                let distance = distances.get(ptr, group);
                distances.0.insert(load.get_result(ctx), distance);
            }
        }
        solver.update_state(ctx, after, |it| it.join(&distances));
        Ok(())
    }

    fn set_to_entry_state(
        _: &DenseForward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: &WriteRef<GroupDistancesLattice>,
    ) {
    }
}

/// Runs the analysis on `root`, whose regions hold the pending operations.
pub fn group_distances(ctx: &Context, root: Ptr<Operation>) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.load(DenseForward::new(GroupDistancesAnalysis));
    solver.initialize_and_run(ctx, root)?;
    Ok(solver)
}

/// The groups of its kind a wait on `token` lets still run, or `None` when it has nothing to
/// wait for.
pub fn groups_left_running(
    solver: &DataflowSolver,
    ctx: &Context,
    wait: Ptr<Operation>,
    token: Value,
) -> Option<usize> {
    let point = ProgramPoint::before_op(ctx, wait);
    let distance = solver
        .lookup_state::<GroupDistancesLattice>(point)
        .and_then(|lattice| lattice.deref().value().0.get(&token).map(|(_, d)| *d))
        .unwrap_or(0);
    (distance != NOTHING_TO_WAIT).then_some(distance as usize)
}

/// Lowers the pending operations: commits to the commit of their kind, and each wait to the
/// wait for its group count. The tokens, and the variables that held them, are removed.
pub struct ResolvePendingPass;

#[pass_name]
impl Pass for ResolvePendingPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let token_ops = token_ops(ctx, op);
        if token_ops.is_empty() {
            return Ok(res);
        }
        res.ir_changed |= IRStatus::Changed;

        let solver = group_distances(ctx, op)?;
        let (waits, holders): (Vec<_>, Vec<_>) = token_ops
            .into_iter()
            .partition(|&op| Operation::get_op::<WaitOp>(op, ctx).is_some());
        let waits = waits
            .into_iter()
            .map(|op| {
                let token = Operation::get_op::<WaitOp>(op, ctx)
                    .expect("partitioned on it")
                    .token(ctx);
                let group = token_group(ctx, token.get_type(ctx)).expect("a wait takes a token");
                (op, group, groups_left_running(&solver, ctx, op, token))
            })
            .collect::<Vec<_>>();
        drop(solver);

        let mut rewriter = IRRewriter::<DummyListener>::default();
        for (wait, group, left_running) in waits {
            match left_running {
                Some(n) => {
                    let lowered = match group {
                        AsyncGroup::Warpgroup => WgmmaWaitGroupOp::new(ctx, n).get_operation(),
                        AsyncGroup::BulkCopy => WaitGroupReadOp::new(ctx, n).get_operation(),
                    };
                    lowered.insert_before(ctx, wait);
                    rewriter.replace_operation(ctx, wait, lowered);
                }
                None => rewriter.erase_operation(ctx, wait),
            }
        }

        // Users before the values they use: the stores and loads of a variable come after it.
        for op in holders.into_iter().rev() {
            if let Some(commit) = Operation::get_op::<CommitOp>(op, ctx) {
                let group = *commit.group(ctx);
                let lowered = match group {
                    AsyncGroup::Warpgroup => WgmmaCommitGroupOp::new(ctx).get_operation(),
                    AsyncGroup::BulkCopy => CommitGroupOp::new(ctx).get_operation(),
                };
                lowered.insert_before(ctx, op);
            }
            rewriter.erase_operation(ctx, op);
        }

        Ok(res)
    }
}

/// The operations that make, hold or use a token, in program order.
fn token_ops(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut ops = Vec::new();
    walk_op(ctx, &mut ops, &WALKCONFIG_ANY, root, |ctx, ops, node| {
        let IRNode::Operation(op) = node else {
            return;
        };
        let dyn_op = op.dyn_op(ctx);
        let holds_token = dyn_op.is::<CommitOp>()
            || dyn_op.is::<ReadyOp>()
            || dyn_op.is::<WaitOp>()
            || dyn_op
                .downcast_ref::<DeclareVariableOp>()
                .is_some_and(|var| token_group(ctx, var.get_result(ctx).get_type(ctx)).is_some())
            || dyn_op
                .downcast_ref::<StoreOp>()
                .is_some_and(|store| token_group(ctx, store.value(ctx).get_type(ctx)).is_some())
            || dyn_op
                .downcast_ref::<LoadOp>()
                .is_some_and(|load| token_group(ctx, load.get_result(ctx).get_type(ctx)).is_some());
        if holds_token {
            ops.push(op);
        }
    });
    ops
}
