use core::{cell::Ref, fmt};

use alloc::format;
use cubecl_ir::{
    interfaces::{
        control_flow::{RegionBranchOpInterface, RegionPredecessor, RegionSuccessor},
        memory_slot::MemoryValue,
    },
    prelude::*,
    rewrite::WALKCONFIG_ANY,
};
use derive_more::{Deref, DerefMut};
use derive_new::new;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    graph::walkers::uninterruptible::immutable::walk_op,
    printable::{self, Printable},
    utils::table::HMap,
};
use smallvec::SmallVec;

use crate::{
    BitSetExt, SparseBitSet,
    analyses::{
        MaybeUninitBitset,
        dataflow_solver::{
            DataflowAnalysis, ProgramPoint, SolverConfig,
            dead_code::{DeadCodeAnalysis, is_block_live},
            dense::{
                DenseBackwardDataflowAnalysis, DenseForwardDataflowAnalysis, DenseLattice,
                LatticeValue,
            },
            sccp::SparseConstantPropagationAnalysis,
            value_dependents::{DependentsLattice, ValueDependentsAnalysis},
            value_numbering::{ValueClassAnalysis, ValueNumberLattice, ValueNumberingAnalysis},
        },
        memory_ssa::MemorySSA,
    },
};

use super::{
    ChangeResult, DataflowSolver, ReadRef, WriteRef,
    dense::{DenseBackward, DenseForward},
};

pub fn compute_lcm_analyses(ctx: &Context, op: Ptr<Operation>) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.load(ValueClassAnalysis::default());
    solver.load(ValueNumberingAnalysis::default());
    solver.load(ComputesAnalysis);
    solver.load(ValueDependentsAnalysis::default());
    solver.initialize_and_run(ctx, op)?;

    let memory_kill_sets = collect_memory_kill_sets(ctx, &solver, op);

    let should_init = |analysis: &dyn DataflowAnalysis| {
        analysis.is::<AvailableAnalysis>() || analysis.is::<AnticipatedAnalysis>()
    };

    solver.load(AvailableAnalysis::default());
    solver.load(AnticipatedAnalysis::new(Anticipated::new(memory_kill_sets)));
    solver.initialize_filtered_and_run(ctx, op, should_init)?;

    Ok(solver)
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct ComputesSet(SparseBitSet);

impl Printable for ComputesSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Computes({{{}}})", set)
    }
}

impl LatticeValue for ComputesSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        let new = ComputesSet(self.union(&rhs.0).into());
        if new == *self {
            ChangeResult::Unchanged
        } else {
            *self = new;
            ChangeResult::Changed
        }
    }
}

pub type Computes = DenseLattice<ComputesSet>;

pub struct ComputesAnalysis;

impl ComputesAnalysis {
    fn update_op(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        op: Ptr<Operation>,
    ) {
        if op.impls::<dyn RegionBranchOpInterface>(ctx) {
            return;
        }

        let mut values = SparseBitSet::new();
        for res in op.deref(ctx).results() {
            let value = solver.get_or_create_for::<Self, ValueNumberLattice>(point, res);
            match value.deref().value().value() {
                Some(value) => values.insert(value as usize),
                None => return,
            }
        }
        let computes_lattice = solver.get_or_create_mut::<Computes>(point);

        solver.update_state(ctx, &computes_lattice, |it| it.meet(&ComputesSet(values)));
    }
}

impl DataflowAnalysis for ComputesAnalysis {
    fn verify(&self, solver: &DataflowSolver, _: &Context, _: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<ValueNumberingAnalysis>()
    }

    fn initialize(
        &mut self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        walk_op(
            ctx,
            &mut (self, solver),
            &WALKCONFIG_ANY,
            root,
            |ctx, (this, solver), node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let point = ProgramPoint::after_op(ctx, op);
                if !is_block_live::<Self>(solver, ctx, point) {
                    return;
                }
                this.update_op(solver, ctx, point, op);
            },
        );
        Ok(())
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if let Some(op) = point.prev_op(ctx) {
            self.update_op(solver, ctx, point, op);
        }
        Ok(())
    }
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct AvailableSet(MaybeUninitBitset);

impl Printable for AvailableSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MaybeUninitBitset::Uninitialized => write!(f, "Available(Unitialized)"),
            MaybeUninitBitset::Initialized(set) => {
                let set = set.iter().map(|i| format!("e{i}")).join(", ");
                write!(f, "Available({{{}}})", set)
            }
        }
    }
}

impl LatticeValue for AvailableSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.intersect(rhs)
    }
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite(rhs)
    }
}

pub type AvailableExpressions = DenseLattice<AvailableSet>;
pub type AvailableAnalysis = DenseForward<Available>;

#[derive(Default)]
pub struct Available;

impl DenseForwardDataflowAnalysis for Available {
    type LatticeValue = AvailableSet;

    fn visit_operation(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let point = ProgramPoint::after_op(ctx, op);
        let computes = solver.get_or_create_for::<Self, Computes>(point, point);

        let before = before.deref();
        let MaybeUninitBitset::Initialized(before_set) = &before.value().0 else {
            return Ok(());
        };

        let union = before_set.union(&**computes.deref().value()).into();
        solver.update_state(ctx, after, |it| it.meet(&AvailableSet(union)));
        Ok(())
    }

    fn set_to_entry_state(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let exit_state = AvailableSet(MaybeUninitBitset::Initialized(Default::default()));
        solver.update_state(ctx, lattice, |it| it.meet(&exit_state));
    }
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct AnticipatedSet(MaybeUninitBitset);

impl Printable for AnticipatedSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MaybeUninitBitset::Uninitialized => write!(f, "Anticipated(Unitialized)"),
            MaybeUninitBitset::Initialized(set) => {
                let set = set.iter().map(|i| format!("e{i}")).join(", ");
                write!(f, "Anticipated({{{}}})", set)
            }
        }
    }
}

impl LatticeValue for AnticipatedSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.unite(rhs)
    }
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.intersect(rhs)
    }
}

pub(crate) fn collect_memory_kill_sets(
    ctx: &Context,
    solver: &DataflowSolver,
    root: Ptr<Operation>,
) -> HMap<MemoryValue, SparseBitSet> {
    let mut out = HMap::new();

    walk_op(
        ctx,
        &mut (solver, &mut out),
        &WALKCONFIG_ANY,
        root,
        |ctx, (solver, out), node| {
            let IRNode::Operation(op) = node else {
                return;
            };
            let mut analyses = solver.analyses();
            let mut memory_ssa = MemorySSA::get_for_nearest_root(ctx, &mut analyses, op).unwrap();

            let Some(mem_use) = memory_ssa.optimized_use(ctx, op) else {
                return;
            };
            let current = out.get(&mem_use);
            let Some(computes) = solver.lookup_state::<Computes>(ProgramPoint::after_op(ctx, op))
            else {
                return;
            };

            // Really annoying workaround because `Ref` does not implement `Clone` for collision
            // reasons, but that makes `Chain` not `Clone` and `reduce` requires `Clone`...
            let dependents = op
                .deref(ctx)
                .results()
                .filter_map(|res| solver.lookup_state::<DependentsLattice>(res))
                .collect::<SmallVec<[_; 4]>>();
            let mut dependents = dependents
                .iter()
                .map(|it| Ref::map(it.deref(), |it| &it.value().0))
                .collect::<SmallVec<[_; 4]>>();
            dependents.push(Ref::map(computes.deref(), |it| &it.value().0));

            let dependents = hi_sparse_bitset::reduce(
                hi_sparse_bitset::ops::Or,
                dependents.iter().map(|it| &**it).chain(current),
            )
            .map(Into::into)
            .unwrap_or_default();
            out.insert(mem_use, dependents);
        },
    );

    out
}

pub type AnticipatedExpressions = DenseLattice<AnticipatedSet>;
pub type AnticipatedAnalysis = DenseBackward<Anticipated>;

/// Simple anticipated without speculation, simpler and should be used on CPU. For GPU, speculative
/// availability should be used (TODO: Actually implement that)
#[derive(new)]
pub struct Anticipated {
    memory_kill_sets: HMap<MemoryValue, SparseBitSet>,
}

impl DenseBackwardDataflowAnalysis for Anticipated {
    type LatticeValue = AnticipatedSet;

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let mut memory_ssa = MemorySSA::get_for_nearest_root(ctx, &mut analyses, op)?;

        let after = after.deref();
        let MaybeUninitBitset::Initialized(after_set) = &after.value().0 else {
            return Ok(());
        };

        let point = ProgramPoint::before_op(ctx, op);
        let after_point = ProgramPoint::after_op(ctx, op);
        let computes = solver.get_or_create_for::<Self, Computes>(point, after_point);
        let computes = computes.deref();
        let before_set = after_set.union(&**computes.value());

        let before_set = if let Some(def) = memory_ssa.op_memory_def(op)
            && let Some(kill_set) = this.memory_kill_sets.get(&def)
        {
            MaybeUninitBitset::Initialized(before_set.difference(kill_set).into())
        } else {
            MaybeUninitBitset::Initialized(before_set.into())
        };
        drop(after);

        solver.update_state(ctx, before, |it| it.join(&AnticipatedSet(before_set)));
        Ok(())
    }

    fn visit_block_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        point: ProgramPoint,
        successor: Ptr<BasicBlock>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let Some(op) = successor.deref(ctx).get_parent_op(ctx) else {
            return this.visit_block_transfer(solver, ctx, block, point, successor, after, before);
        };
        let mut analyses = solver.analyses();
        let mut memory_ssa = MemorySSA::get_for_nearest_root(ctx, &mut analyses, op)?;

        let Some(def) = memory_ssa.block_memory_def(successor) else {
            return this.visit_block_transfer(solver, ctx, block, point, successor, after, before);
        };
        let Some(kill_set) = this.memory_kill_sets.get(&def) else {
            return this.visit_block_transfer(solver, ctx, block, point, successor, after, before);
        };

        let new_value = after.deref().value().raw_difference(kill_set);
        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(new_value)));

        Ok(())
    }

    fn visit_region_branch_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        from: RegionPredecessor,
        to: RegionSuccessor,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let mut memory_ssa =
            MemorySSA::get_for_nearest_root(ctx, &mut analyses, branch.get_operation())?;

        let def = match to {
            RegionSuccessor::Region(region)
                if let Some(entry) = region.deref(ctx).get_entry_block() =>
            {
                memory_ssa.block_memory_def(entry)
            }
            RegionSuccessor::AfterOp => memory_ssa.op_memory_def(branch.get_operation()),
            _ => None,
        };
        let Some(def) = def else {
            return this.visit_region_branch_control_flow_transfer(
                solver, ctx, branch, from, to, after, before,
            );
        };
        let Some(kill_set) = this.memory_kill_sets.get(&def) else {
            return this.visit_region_branch_control_flow_transfer(
                solver, ctx, branch, from, to, after, before,
            );
        };

        let new_value = after.deref().value().raw_difference(kill_set);
        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(new_value)));
        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let exit_state = AnticipatedSet(MaybeUninitBitset::Initialized(Default::default()));
        solver.update_state(ctx, lattice, |it| it.meet(&exit_state));
    }
}
