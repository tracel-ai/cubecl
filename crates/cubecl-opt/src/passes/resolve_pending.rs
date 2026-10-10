//! Resolves the waits on commit group tokens into waits on group counts.
//!
//! A wait on a token becomes a wait that lets `n` groups of the token's kind still run, where `n`
//! is the fewest groups committed after the token's group on any path from its commit to the
//! wait. Groups of one kind complete in commit order, so once at most `n` of the newest are left
//! running, the token's group is done on every path. Each token is tracked through the variables
//! that hold it; a wait on a token the analysis lost track of waits for every group.

use core::fmt;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_ir::{
    dialect::{
        branch::ReturnOp,
        memory::{DeclareVariableOp, LoadOp, StoreOp},
        pending::{CommitOp, ReadyOp, WaitOp},
    },
    prelude::*,
    rewrite::WALKCONFIG_ANY,
    types::{
        PointerType,
        pending::{AsyncGroup, GroupTokenType, WaitUntil},
    },
};
use itertools::Itertools;
use pliron::{
    graph::walkers::uninterruptible::immutable::walk_op,
    irbuild::listener::DummyListener,
    printable::{self, Printable},
    verify_err,
};
use thiserror::Error;

use crate::analyses::dataflow_solver::{
    ChangeResult, DataflowSolver, ProgramPoint, ReadRef, SolverConfig, WriteRef,
    dead_code::DeadCodeAnalysis,
    dense::{DenseForward, DenseForwardDataflowAnalysis, DenseLattice, LatticeValue},
    sccp::SparseConstantPropagationAnalysis,
};

#[derive(Error, Debug)]
enum ResolvePendingError {
    #[error(
        "a `Pending` is used by `{0}`, and the wait counts can't follow it there. Hold it in \
         a variable, or keep a fixed number in flight in a `Sequence` indexed in a loop unrolled \
         over the stages"
    )]
    UnsupportedUse(String),
}

/// How far a token's group is behind the newest of its kind. Every count is closer than
/// [`Distance::Ready`], so merging paths keeps the smaller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Distance {
    /// Groups of the same kind committed after the token's.
    Commits(u32),
    /// Nothing to wait for.
    Ready,
}

impl Distance {
    fn after_commit(self) -> Self {
        match self {
            Distance::Commits(commits) => Distance::Commits(commits + 1),
            Distance::Ready => Distance::Ready,
        }
    }
}

/// What the analysis knows of a token, or of the variable holding one.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TokenState {
    group: AsyncGroup,
    distance: Distance,
}

/// An operation that makes, holds or uses a token.
enum TokenOp {
    Commit {
        group: AsyncGroup,
        token: Value,
    },
    Ready {
        group: AsyncGroup,
        token: Value,
    },
    Wait {
        group: AsyncGroup,
        token: Value,
        until: WaitUntil,
    },
    Declare {
        slot: Value,
    },
    Store {
        group: AsyncGroup,
        slot: Value,
        token: Value,
    },
    Load {
        group: AsyncGroup,
        slot: Value,
        token: Value,
    },
}

impl TokenOp {
    fn new(ctx: &Context, op: Ptr<Operation>) -> Option<Self> {
        let op = op.dyn_op(ctx);
        if let Some(commit) = op.downcast_ref::<CommitOp>() {
            let group = *commit.group(ctx);
            return Some(TokenOp::Commit {
                group,
                token: commit.get_result(ctx),
            });
        }
        if let Some(ready) = op.downcast_ref::<ReadyOp>() {
            let group = *ready.group(ctx);
            return Some(TokenOp::Ready {
                group,
                token: ready.get_result(ctx),
            });
        }
        if let Some(wait) = op.downcast_ref::<WaitOp>() {
            let token = wait.token(ctx);
            let until = *wait.until(ctx);
            let group = token_group(ctx, token.get_type(ctx))?;
            return Some(TokenOp::Wait {
                group,
                token,
                until,
            });
        }
        if let Some(declare) = op.downcast_ref::<DeclareVariableOp>() {
            let slot = declare.get_result(ctx);
            token_group(ctx, slot.get_type(ctx))?;
            return Some(TokenOp::Declare { slot });
        }
        if let Some(store) = op.downcast_ref::<StoreOp>() {
            let token = store.value(ctx);
            let group = token_group(ctx, token.get_type(ctx))?;
            let slot = store.ptr(ctx);
            return Some(TokenOp::Store { group, slot, token });
        }
        if let Some(load) = op.downcast_ref::<LoadOp>() {
            let token = load.get_result(ctx);
            let group = token_group(ctx, token.get_type(ctx))?;
            let slot = load.ptr(ctx);
            return Some(TokenOp::Load { group, slot, token });
        }
        None
    }

    /// The token or variable this operation defines, which only other token operations may use.
    fn defines(&self) -> Option<Value> {
        match *self {
            TokenOp::Commit { token, .. }
            | TokenOp::Ready { token, .. }
            | TokenOp::Load { token, .. } => Some(token),
            TokenOp::Declare { slot } => Some(slot),
            TokenOp::Wait { .. } | TokenOp::Store { .. } => None,
        }
    }
}

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

/// For each token, and each variable holding one, the fewest groups of its kind committed since
/// its own on any path to this point.
#[derive(Default, PartialEq, Clone)]
struct GroupDistances(HashMap<Value, TokenState>);

impl GroupDistances {
    /// What is known of `slot`. A token the analysis lost track of may have just been committed.
    fn get(&self, slot: Value, group: AsyncGroup) -> TokenState {
        self.0.get(&slot).copied().unwrap_or(TokenState {
            group,
            distance: Distance::Commits(0),
        })
    }

    fn apply(&mut self, op: &TokenOp) {
        match *op {
            TokenOp::Commit { group, token } => {
                for state in self.0.values_mut().filter(|state| state.group == group) {
                    state.distance = state.distance.after_commit();
                }
                let distance = Distance::Commits(0);
                self.0.insert(token, TokenState { group, distance });
            }
            TokenOp::Ready { group, token } => {
                let distance = Distance::Ready;
                self.0.insert(token, TokenState { group, distance });
            }
            TokenOp::Store { group, slot, token } => {
                self.0.insert(slot, self.get(token, group));
            }
            TokenOp::Load { group, slot, token } => {
                self.0.insert(token, self.get(slot, group));
            }
            TokenOp::Wait { .. } | TokenOp::Declare { .. } => {}
        }
    }
}

impl Printable for GroupDistances {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self
            .0
            .iter()
            .map(|(slot, state)| format!("{}: {:?}", slot.disp(ctx), state.distance))
            .sorted()
            .join(", ");
        write!(f, "GroupDistances({{{entries}}})")
    }
}

impl LatticeValue for GroupDistances {
    /// Paths merge to the fewest commits either took.
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        let mut change = ChangeResult::Unchanged;
        for (slot, state) in &rhs.0 {
            let entry = self.0.entry(*slot).or_insert_with(|| {
                change = ChangeResult::Changed;
                *state
            });
            if state.distance < entry.distance {
                entry.distance = state.distance;
                change = ChangeResult::Changed;
            }
        }
        change
    }
}

type GroupDistancesLattice = DenseLattice<GroupDistances>;

/// Tracks [`GroupDistances`] forward through the function.
#[derive(Default)]
struct GroupDistancesAnalysis;

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
        let Some(token_op) = TokenOp::new(ctx, op) else {
            let before = before.deref();
            solver.update_state(ctx, after, |it| it.join(before.value()));
            return Ok(());
        };
        let mut distances = before.deref().value().clone();
        distances.apply(&token_op);
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

/// The group counts the waits of a function resolve to.
struct WaitCounts(DataflowSolver);

impl WaitCounts {
    fn new(ctx: &Context, root: Ptr<Operation>) -> Result<Self> {
        let mut solver = DataflowSolver::new(SolverConfig::default());
        solver.load(DeadCodeAnalysis::default());
        solver.load(SparseConstantPropagationAnalysis::default());
        solver.load(DenseForward::new(GroupDistancesAnalysis));
        solver.initialize_and_run(ctx, root)?;
        Ok(Self(solver))
    }

    /// The groups of its kind the wait on `token` lets still run, or `None` when it has nothing
    /// to wait for.
    fn max_pending(
        &self,
        ctx: &Context,
        wait: Ptr<Operation>,
        group: AsyncGroup,
        token: Value,
    ) -> Option<usize> {
        let point = ProgramPoint::before_op(ctx, wait);
        let state = match self.0.lookup_state::<GroupDistancesLattice>(point) {
            Some(lattice) => lattice.deref().value().get(token, group),
            None => GroupDistances::default().get(token, group),
        };
        match state.distance {
            Distance::Commits(commits) => Some(commits as usize),
            Distance::Ready => None,
        }
    }
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
        check_uses(ctx, &token_ops)?;
        res.ir_changed |= IRStatus::Changed;

        let wait_counts = WaitCounts::new(ctx, op)?;
        let lowered_waits = token_ops
            .iter()
            .filter_map(|(op, token_op)| match *token_op {
                TokenOp::Wait {
                    group,
                    token,
                    until,
                } => {
                    let max_pending = wait_counts.max_pending(ctx, *op, group, token);
                    Some((*op, group, until, max_pending))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        drop(wait_counts);

        let mut rewriter = IRRewriter::<DummyListener>::default();
        for (wait, group, until, max_pending) in lowered_waits {
            match max_pending {
                Some(max_pending) => {
                    let lowered = group.wait_op(ctx, max_pending, until);
                    lowered.insert_before(ctx, wait);
                    rewriter.replace_operation(ctx, wait, lowered);
                }
                None => rewriter.erase_operation(ctx, wait),
            }
        }

        // A bulk copy reads shared memory until its group completes, and the cube's shared memory
        // goes to the next cube once it retires: a store left unwaited is waited for at each
        // return, wherever a bulk group was committed.
        let commits_bulk = token_ops.iter().any(|(_, token_op)| {
            matches!(
                token_op,
                TokenOp::Commit {
                    group: AsyncGroup::BulkCopy,
                    ..
                }
            )
        });
        if commits_bulk {
            for ret in returns(ctx, op) {
                let wait = AsyncGroup::BulkCopy.wait_op(ctx, 0, WaitUntil::Released);
                wait.insert_before(ctx, ret);
            }
        }

        // Users before the values they use: the stores and loads of a variable come after it.
        for (op, token_op) in token_ops.into_iter().rev() {
            match token_op {
                TokenOp::Wait { .. } => continue,
                TokenOp::Commit { group, .. } => group.commit_op(ctx).insert_before(ctx, op),
                _ => {}
            }
            rewriter.erase_operation(ctx, op);
        }

        Ok(res)
    }
}

/// The operations that make, hold or use a token, in program order.
fn token_ops(ctx: &Context, root: Ptr<Operation>) -> Vec<(Ptr<Operation>, TokenOp)> {
    let mut ops = Vec::new();
    walk_op(ctx, &mut ops, &WALKCONFIG_ANY, root, |ctx, ops, node| {
        if let IRNode::Operation(op) = node
            && let Some(token_op) = TokenOp::new(ctx, op)
        {
            ops.push((op, token_op));
        }
    });
    ops
}

/// The returns of the function `root`.
fn returns(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut returns = Vec::new();
    walk_op(
        ctx,
        &mut returns,
        &WALKCONFIG_ANY,
        root,
        |ctx, returns, node| {
            if let IRNode::Operation(op) = node
                && op.dyn_op(ctx).downcast_ref::<ReturnOp>().is_some()
            {
                returns.push(op);
            }
        },
    );
    returns
}

/// Checks that only token operations use a token or a variable holding one, so removing them
/// all leaves no dangling use.
fn check_uses(ctx: &Context, token_ops: &[(Ptr<Operation>, TokenOp)]) -> Result<()> {
    let users = token_ops.iter().map(|(op, _)| *op).collect::<HashSet<_>>();
    for value in token_ops
        .iter()
        .filter_map(|(_, token_op)| token_op.defines())
    {
        for r#use in value.uses(ctx) {
            let user = r#use.user_op();
            if !users.contains(&user) {
                let loc = user.deref(ctx).loc();
                let name = Operation::get_opid(user, ctx).to_string();
                return verify_err!(loc, ResolvePendingError::UnsupportedUse(name));
            }
        }
    }
    Ok(())
}
