//! Partial Redundancy Elimination, using techniques from:
//! * [GVN-PRE; Vandrunen, Thomas & Hosking, Antony. (2004). Value-Based Partial Redundancy
//!   Elimination.](https://doi.org/10.1007/978-3-540-24723-4_12)
//! * LLVM's `MemorySSA`, adapted to the region-based IR and expanded with interprocedural support
//! * Speculative Code Motion, loosely based on
//!   [Fukuhara, Junji and Munehiro Takimoto. “Branch Divergence Reduction Based on Code Motion.” J.
//!   Inf. Process. 28 (2020): 302-309.](https://doi.org/10.2197/ipsjjip.28.302)
//!

use core::{cell::Ref, fmt, ops::Deref};

use alloc::format;
use cubecl_ir::{
    dialect::{BlockPtrExt, RegionPtrExt},
    interfaces::{
        control_flow::{
            CallableOpInterface, HoistableRegionOpInterface, RegionBranchOpInterface,
            RegionPredecessor, RegionSuccessor,
        },
        memory_slot::MemoryValue,
        uniformity::{
            UniformBranchOpInterface, UniformRegionOpInterface, UniformRegionTerminatorOpInterface,
            Uniformity,
        },
    },
    prelude::*,
    rewrite::WALKCONFIG_ANY,
};
use derive_more::{Deref, DerefMut};
use derive_new::new;
use hi_sparse_bitset::ops;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    graph::{ControlFlowGraph, walkers::uninterruptible::immutable::walk_op},
    printable::{self, Printable},
    utils::table::{HMap, IMap},
};
use smallvec::SmallVec;

use crate::{
    BitSetExt, BitsetLatticeExt, SparseBitSet,
    analyses::{
        MaybeUninitBitset,
        dataflow_solver::{
            DataflowAnalysis, ProgramPoint, SmallPtrVec, SolverConfig,
            dead_code::{Executable, is_block_live},
            dense::{
                DenseBackwardDataflowAnalysis, DenseForwardDataflowAnalysis, DenseLattice,
                LatticeValue,
            },
            speculatability::{NotSpeculatable, SpeculatableAnalysis},
            value_dependents::{DependentsLattice, ValueDependentsAnalysis},
            value_numbering::{ValueNumberLattice, ValueNumberingAnalysis},
            value_uniformity::{DynamicUniformityAnalysis, DynamicUniformityLattice},
        },
        memory_ssa::MemorySSA,
    },
};

use super::{
    ChangeResult, DataflowSolver, ReadRef, WriteRef,
    dense::{DenseBackward, DenseForward},
};

/// Mode for computing anticipated sets
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AnticipatedMode {
    /// Expressions are only anticipated if they're computed on all paths, should be used on CPU
    Classical,
    /// Speculative Code Motion, uses `Classical` for uniform branches but anticipates expressions
    /// computed on *any* path for divergent branches, as long as they're speculatable.
    Speculative,
}

/// Mode for computing insertions
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InsertMode {
    /// Lazy insertions minimize register lifetime but can cause divergence and code size to increase.
    /// Should probably be used on CPU.
    Lazy,
    /// Approximation of Sparse Code Motion, can increase register lifetime but reduces divergence
    /// and code size. Should be used on GPU.
    Sparse,
}

pub fn compute_lcm_analyses(
    ctx: &Context,
    op: Ptr<Operation>,
    mode: AnticipatedMode,
) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(ComputesAnalysis);
    solver.load(ValueDependentsAnalysis::default());
    solver.initialize_and_run(ctx, op)?;

    let finished_analyses = solver.loaded_analyses().to_vec();
    let memory_kill_sets = collect_memory_kill_sets(ctx, &solver, op);

    let should_init =
        |analysis: &dyn DataflowAnalysis| !finished_analyses.contains(&analysis.type_id());

    if mode == AnticipatedMode::Speculative {
        solver.load(SpeculatableAnalysis::default());
        solver.load(DynamicUniformityAnalysis::default());
    }

    solver.load(AvailableAnalysis::default());
    solver.load(AnticipatedBackwards::new(memory_kill_sets.clone()));
    solver.initialize_filtered_and_run(ctx, op, should_init)?;

    if mode == AnticipatedMode::Speculative {
        // Standard anticipated runs a most fixpoint starting with `U` and intersecting,
        // so it cannot converge together with the speculative anticipated which runs a least fixpoint
        // starting with ∅ and unioning. So we split it into the seed rounds that only seed speculative
        // anticipation and then reset and re-run the standard anticipated with the speculatively
        // transferred sets merged into the transfer function. This can require more than one iteration
        // to converge because speculative expressions may become strictly anticipated in an outer branch
        // and then become speculatively available on another edge.
        let mut n_iters = 0;
        let rerun_antic = |analysis: &dyn DataflowAnalysis| analysis.is::<AnticipatedAnalysis>();
        while seed_speculative(ctx, &solver, op, &memory_kill_sets)? == ChangeResult::Changed {
            solver.clear_states_of_type::<Anticipated>();
            solver.initialize_filtered_and_run(ctx, op, rerun_antic)?;
            // Safety shutoff because I don't have a mathematical proof of convergence
            n_iters += 1;
            if n_iters > 10 {
                break;
            }
        }
    }

    Ok(solver)
}

pub fn compute_pruned_insertion_points(
    ctx: &Context,
    op: Ptr<Operation>,
    solver: &mut DataflowSolver,
    mode: InsertMode,
) -> Result<IMap<ProgramPoint, SparseBitSet>> {
    let finished_analyses = solver.loaded_analyses().to_vec();
    let should_init =
        |analysis: &dyn DataflowAnalysis| !finished_analyses.contains(&analysis.type_id());

    solver.load(InsertedAnalysis::default());
    if mode == InsertMode::Sparse {
        solver.load(SparseInsertedAnalysis::default());
    }
    solver.initialize_filtered_and_run(ctx, op, should_init)?;

    // Rerun available because it can't grow but we need to prune based on new availability facts
    // post insertion
    let rerun_avail = |analysis: &dyn DataflowAnalysis| analysis.is::<AvailableAnalysis>();
    // Don't want to rerun other stuff
    for computes in solver.states_of_type::<Computes>() {
        computes.deref().clear_dependents();
    }
    solver.clear_states_of_type::<Available>();
    solver.initialize_filtered_and_run(ctx, op, rerun_avail)?;

    let mut out = IMap::default();
    let inserted = solver.states_of_type::<Inserted>();
    for inserted in inserted {
        let point = inserted.deref().anchor();
        let mut inserted = inserted.deref().value().0.clone();
        // Clean up inserts that were later made redundant
        if let Some(op) = point.prev_op(ctx)
            && let Some(avail) = solver.lookup_state::<Available>(ProgramPoint::before_op(ctx, op))
        {
            inserted = inserted.difference(avail.deref().value().0.unwrap()).into();
        }
        if !inserted.is_empty() {
            out.insert(point, inserted);
        }
    }
    Ok(out)
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct ComputesSet(pub SparseBitSet);

impl Printable for ComputesSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Computes({{{}}})", set)
    }
}

impl LatticeValue for ComputesSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.0.unite_changed(&rhs.0)
    }

    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.0.unite_changed(&rhs.0)
    }
}

pub type Computes = DenseLattice<ComputesSet>;

#[derive(new, Default)]
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
    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(ValueNumberingAnalysis::default);
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
pub struct AvailableSet(pub MaybeUninitBitset);

impl Printable for AvailableSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Available({{{}}})", set)
    }
}

impl LatticeValue for AvailableSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.intersect(rhs)
    }
}

pub type Available = DenseLattice<AvailableSet>;
pub type AvailableAnalysis = DenseForward<AvailableForward>;

#[derive(Default)]
pub struct AvailableForward;

impl AvailableForward {
    fn extend_from_values(
        solver: &DataflowSolver,
        point: ProgramPoint,
        set: &mut SparseBitSet,
        values: impl IntoIterator<Item = Value>,
    ) {
        for arg in values {
            let value_number =
                solver.get_or_create_for::<AvailableAnalysis, ValueNumberLattice>(point, arg);
            if let Some(value_number) = value_number.deref().value().value() {
                set.insert(value_number as usize);
            }
        }
    }
}

impl DenseForwardDataflowAnalysis for AvailableForward {
    type LatticeValue = AvailableSet;

    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(ComputesAnalysis::new);
    }

    fn visit_operation(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let point = ProgramPoint::after_op(ctx, op);
        let computes = solver.get_or_create_for::<AvailableAnalysis, Computes>(point, point);

        if after == before {
            Self::set_to_entry_state(this, solver, ctx, after);
            return Ok(());
        }

        let before = before.deref();
        let MaybeUninitBitset::Initialized(before) = &before.value().0 else {
            return Ok(());
        };

        let union = before.union(&computes.deref().value().0).into();
        solver.update_state(ctx, after, |it| it.join(&AvailableSet(union)));
        Ok(())
    }

    fn visit_region_branch_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        point: ProgramPoint,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let branch_op = branch.get_operation();
        if branch_op.impls::<dyn HoistableRegionOpInterface>(ctx) {
            Self::visit_control_flow_transfer(this, solver, ctx, predecessors, after);
        } else if branch_op.impls::<dyn IsolatedFromAboveInterface>(ctx)
            && point != ProgramPoint::after_op(ctx, branch_op)
        {
            let mut before = SparseBitSet::new();
            let args = point.block().unwrap().arguments(ctx);
            Self::extend_from_values(solver, point, &mut before, args);
            solver.update_state(ctx, after, |it| {
                it.join(&AvailableSet(MaybeUninitBitset::Initialized(before)))
            });
        } else {
            // Nested regions are discarded and only things before the region are made available
            // because we can't thread new values out of them
            let before =
                this.get_lattice_for(solver, point, ProgramPoint::before_op(ctx, branch_op));
            let before = before.deref();
            let MaybeUninitBitset::Initialized(mut before) = before.value().0.clone() else {
                return;
            };
            if point == ProgramPoint::after_op(ctx, branch_op) {
                Self::extend_from_values(solver, point, &mut before, branch_op.results(ctx));
            } else if point.is_block_start(ctx) {
                let args = point.block().unwrap().arguments(ctx);
                Self::extend_from_values(solver, point, &mut before, args);
            }
            solver.update_state(ctx, after, |it| {
                it.join(&AvailableSet(MaybeUninitBitset::Initialized(before)))
            });
        }
    }

    fn visit_control_flow_transfer(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        for predecessor in predecessors {
            let pred = predecessor.deref();
            let MaybeUninitBitset::Initialized(pred) = &pred.value().0 else {
                continue;
            };

            let point = predecessor.deref().anchor();
            let computes = solver.get_or_create_for::<AvailableAnalysis, Computes>(point, point);
            let union = pred.union(&computes.deref().value().0).into();
            solver.update_state(ctx, after, |it| it.join(&AvailableSet(union)));
        }
    }

    fn visit_callee_control_flow_transfer(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
        _: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        // Maybe add callable interface later that allows adding args
        let Some(region) = callable.callable_region(ctx) else {
            return;
        };
        let Some(entry) = region.deref(ctx).get_entry_block() else {
            return;
        };
        let point = ProgramPoint::at_block_start(ctx, entry);
        let mut before = SparseBitSet::new();
        let args = callable.callable_region(ctx).unwrap().arguments(ctx);
        Self::extend_from_values(solver, point, &mut before, args);
        solver.update_state(ctx, after, |it| {
            it.join(&AvailableSet(MaybeUninitBitset::Initialized(before)))
        });
    }

    fn visit_return_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        _: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let call_op = call.get_operation();
        let point = ProgramPoint::after_op(ctx, call_op);
        // Maybe add call interface later that allows adding args
        let before = this.get_lattice_for(solver, point, ProgramPoint::before_op(ctx, call_op));
        let before = before.deref();
        let MaybeUninitBitset::Initialized(mut before) = before.value().0.clone() else {
            return;
        };
        Self::extend_from_values(solver, point, &mut before, call_op.deref(ctx).results());
        solver.update_state(ctx, after, |it| {
            it.join(&AvailableSet(MaybeUninitBitset::Initialized(before)))
        });
    }

    fn set_to_entry_state(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| {
            it.join(&AvailableSet(MaybeUninitBitset::empty_init()))
        });
    }
}

pub(crate) fn collect_memory_kill_sets(
    ctx: &Context,
    solver: &DataflowSolver,
    root: Ptr<Operation>,
) -> HMap<MemoryValue, SparseBitSet> {
    solver
        .require_loaded::<ComputesAnalysis>()
        .expect("Missing `Computes` state");
    solver
        .require_loaded::<ValueDependentsAnalysis>()
        .expect("Missing `Dependents` state");

    let mut out = HMap::new();

    walk_op(
        ctx,
        &mut (solver, &mut out, root),
        &WALKCONFIG_ANY,
        root,
        |ctx, (solver, out, root), node| {
            let IRNode::Operation(op) = node else {
                return;
            };
            let mut analyses = solver.analyses();
            let mut memory_ssa = analyses.get_analysis_mut::<MemorySSA>(*root, ctx).unwrap();

            let Some(mem_use) = memory_ssa.optimized_use(ctx, op) else {
                return;
            };
            let current = out.get(&mem_use);
            let Some(computes) = solver.lookup_state::<Computes>(ProgramPoint::after_op(ctx, op))
            else {
                return;
            };

            let mut dependents = op_transitive_kill_set(solver, &computes, op.deref(ctx).results());
            if let Some(current) = current {
                dependents.unite(current);
            }

            out.insert(mem_use, dependents);
        },
    );

    out
}

pub(super) fn op_transitive_kill_set(
    solver: &DataflowSolver,
    computes: &ReadRef<Computes>,
    values: impl Iterator<Item = Value>,
) -> SparseBitSet {
    // Really annoying workaround because `Ref` does not implement `Clone` for collision
    // reasons, but that makes `Chain` not `Clone` and `reduce` requires `Clone`...
    let dependents = values
        .filter_map(|res| solver.lookup_state::<DependentsLattice>(res))
        .collect::<SmallVec<[_; 4]>>();
    let mut dependents = dependents
        .iter()
        .map(|it| Ref::map(it.deref(), |it| &it.value().0))
        .collect::<SmallVec<[_; 4]>>();
    dependents.push(Ref::map(computes.deref(), |it| &it.value().0));

    hi_sparse_bitset::reduce(hi_sparse_bitset::ops::Or, dependents.iter().map(|it| &**it))
        .map(Into::into)
        .unwrap_or_default()
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct AnticipatedSet(MaybeUninitBitset);

impl Printable for AnticipatedSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Anticipated({{{}}})", set)
    }
}

impl LatticeValue for AnticipatedSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.intersect(rhs)
    }
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct SpeculativelyAnticipatedSet(SparseBitSet);

impl Printable for SpeculativelyAnticipatedSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "SpeculativelyAnticipated({{{}}})", set)
    }
}

impl LatticeValue for SpeculativelyAnticipatedSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type SpeculativelyAnticipated = DenseLattice<SpeculativelyAnticipatedSet>;
pub type Anticipated = DenseLattice<AnticipatedSet>;
pub type AnticipatedAnalysis = DenseBackward<AnticipatedBackwards>;

pub struct AnticipatedBackwards {
    memory_kill_sets: HMap<MemoryValue, SparseBitSet>,
}

impl AnticipatedBackwards {
    pub fn new(memory_kill_sets: HMap<MemoryValue, SparseBitSet>) -> AnticipatedAnalysis {
        DenseBackward::new(Self { memory_kill_sets })
    }
}

impl DenseBackwardDataflowAnalysis for AnticipatedBackwards {
    type LatticeValue = AnticipatedSet;

    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(ComputesAnalysis::new);
    }

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let memory_ssa = analyses.get_analysis::<MemorySSA>(solver.root().unwrap(), ctx)?;

        if after == before {
            Self::set_to_exit_state(this, solver, ctx, before);
            return Ok(());
        }

        let after = after.deref();
        let MaybeUninitBitset::Initialized(after) = &after.value().0 else {
            return Ok(());
        };

        let point = ProgramPoint::before_op(ctx, op);
        let after_point = ProgramPoint::after_op(ctx, op);
        let computes =
            solver.get_or_create_for::<AnticipatedAnalysis, Computes>(point, after_point);
        let computes = computes.deref();
        let before_set = after.union(&**computes.value());

        let before_set = if let Some(def) = memory_ssa.op_memory_def(op)
            && let Some(kill_set) = this.memory_kill_sets.get(&def)
        {
            before_set.difference(kill_set).into()
        } else {
            before_set.into()
        };
        let _ = after;

        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(before_set)));
        Ok(())
    }

    fn visit_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        succs: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let memory_ssa = analyses.get_analysis::<MemorySSA>(solver.root().unwrap(), ctx)?;

        let speculative = solver.get_or_create::<SpeculativelyAnticipated>(point);
        let speculative = speculative.deref();
        let speculative = &speculative.value().0;

        for succ in succs {
            let point = succ.deref().anchor();
            let def = match point.prev_op(ctx) {
                Some(op) => memory_ssa.op_memory_def(op),
                None => memory_ssa.block_memory_def(point.block().unwrap()),
            };
            let succ = succ.deref();
            let MaybeUninitBitset::Initialized(succ) = &succ.value().0 else {
                continue;
            };
            if let Some(def) = def
                && let Some(kill_set) = this.memory_kill_sets.get(&def)
            {
                let succ = succ.union(speculative).difference(kill_set).into();
                solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(succ)));
            } else {
                let succ = succ.union(speculative).into();
                solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(succ)));
            }
        }

        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| {
            it.meet(&AnticipatedSet(MaybeUninitBitset::empty_init()))
        });
    }
}

fn seed_speculative(
    ctx: &Context,
    solver: &DataflowSolver,
    op: Ptr<Operation>,
    memory_kill_sets: &HMap<MemoryValue, SparseBitSet>,
) -> Result<ChangeResult> {
    struct Updater<'a> {
        memory_ssa: Ref<'a, MemorySSA>,
        memory_kill_sets: &'a HMap<MemoryValue, SparseBitSet>,
    }

    impl Updater<'_> {
        fn update_from(
            &self,
            ctx: &Context,
            solver: &DataflowSolver,
            before: &WriteRef<SpeculativelyAnticipated>,
            succ_point: ProgramPoint,
        ) -> ChangeResult {
            let not_speculatable = solver.lookup_state::<NotSpeculatable>(succ_point).unwrap();
            let not_speculatable = not_speculatable.deref();
            let antic = solver.lookup_state::<Anticipated>(succ_point).unwrap();
            let antic = antic.deref();
            let antic = antic.value().0.unwrap();
            let spec_antic_succ = antic.difference(&not_speculatable.value().0);

            let def = match succ_point.is_block_start(ctx) {
                true => self
                    .memory_ssa
                    .block_memory_def(succ_point.block().unwrap()),
                false => self
                    .memory_ssa
                    .op_memory_def(succ_point.prev_op(ctx).unwrap()),
            };

            let spec_antic_succ = if let Some(def) = def
                && let Some(kill) = self.memory_kill_sets.get(&def)
            {
                spec_antic_succ.difference(kill).into()
            } else {
                spec_antic_succ.into()
            };
            let mut changed = ChangeResult::Unchanged;
            solver.update_state(ctx, before, |it| {
                changed = it.meet(&SpeculativelyAnticipatedSet(spec_antic_succ));
                changed
            });
            changed
        }
    }

    let mut analyses = solver.analyses();
    let memory_ssa = analyses.get_analysis::<MemorySSA>(op, ctx)?;
    let updater = Updater {
        memory_ssa,
        memory_kill_sets,
    };

    let mut changed = ChangeResult::Unchanged;
    walk_op(
        ctx,
        &mut (solver, &updater, &mut changed),
        &WALKCONFIG_ANY,
        op,
        |ctx, (solver, updater, changed), op| {
            let IRNode::Operation(op) = op else {
                return;
            };
            let Some(block) = op.deref(ctx).get_parent_block() else {
                return;
            };
            let executable =
                solver.lookup_state::<Executable>(ProgramPoint::at_block_start(ctx, block).into());
            if let Some(executable) = executable
                && !executable.deref().is_live()
            {
                return;
            }
            if let Some(branch) = op.cast::<dyn UniformBranchOpInterface>(ctx)
                && is_branch_divergent(ctx, solver, &*branch)
            {
                let before = ProgramPoint::at_block_end(ctx, block);
                let spec_antic = solver.get_or_create_mut::<SpeculativelyAnticipated>(before);
                for succ in op.deref(ctx).successors() {
                    let succ_point = ProgramPoint::at_block_start(ctx, succ);
                    let executable = solver.lookup_state::<Executable>(succ_point.into());
                    if let Some(executable) = executable
                        && !executable.deref().is_live()
                    {
                        continue;
                    }
                    **changed |= updater.update_from(ctx, solver, &spec_antic, succ_point);
                }
                return;
            }
            let (before, successors) = if let Some(region_op) =
                op.cast::<dyn UniformRegionOpInterface>(ctx)
                && is_region_op_divergent(ctx, solver, &*region_op)
            {
                let before = ProgramPoint::before_op(ctx, op);
                let successors = region_op.successor_regions(ctx, RegionPredecessor::Parent);
                (before, successors)
            } else if let Some(term) = op.cast::<dyn UniformRegionTerminatorOpInterface>(ctx)
                && is_region_terminator_divergent(ctx, solver, &*term)
            {
                let before = ProgramPoint::after_op(ctx, op);
                let successors = term.all_successor_regions(ctx);
                (before, successors)
            } else {
                return;
            };

            let spec_antic = solver.get_or_create_mut::<SpeculativelyAnticipated>(before);
            for succ in successors {
                let succ_point = match succ {
                    RegionSuccessor::Region(region) if let Some(entry) = region.entry_node(ctx) => {
                        ProgramPoint::at_block_start(ctx, entry)
                    }
                    RegionSuccessor::Region(_) | RegionSuccessor::AfterOp => {
                        ProgramPoint::after_op(ctx, op)
                    }
                };
                let executable = solver.lookup_state::<Executable>(succ_point.into());
                if let Some(executable) = executable
                    && !executable.deref().is_live()
                {
                    continue;
                }
                **changed |= updater.update_from(ctx, solver, &spec_antic, succ_point);
            }
        },
    );
    Ok(changed)
}

fn is_branch_divergent(
    ctx: &Context,
    solver: &DataflowSolver,
    branch: &dyn UniformBranchOpInterface,
) -> bool {
    let branch_op = branch.get_operation();
    let mut operands = SmallPtrVec::new();
    for opd in branch_op.deref(ctx).operands() {
        let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
            panic!("Dynamic uniformity should be converged before anticipated sets are run")
        };
        operands.push(lattice.deref().value().0);
    }
    branch.successor_uniformity(ctx, &operands) <= Uniformity::None
}

fn is_region_op_divergent(
    ctx: &Context,
    solver: &DataflowSolver,
    region_op: &dyn UniformRegionOpInterface,
) -> bool {
    let op = region_op.get_operation();
    let mut operands = SmallPtrVec::new();
    for opd in op.deref(ctx).operands() {
        let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
            panic!("Dynamic uniformity should be converged before anticipated sets are run")
        };
        operands.push(lattice.deref().value().0);
    }
    region_op.entry_successor_region_uniformity(ctx, &operands) <= Uniformity::None
}

fn is_region_terminator_divergent(
    ctx: &Context,
    solver: &DataflowSolver,
    term: &dyn UniformRegionTerminatorOpInterface,
) -> bool {
    let mut operands = SmallPtrVec::new();
    for opd in term.get_operation().deref(ctx).operands() {
        let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
            panic!("Dynamic uniformity should be converged before anticipated sets are run")
        };
        operands.push(lattice.deref().value().0);
    }
    term.successor_region_uniformity(ctx, &operands) <= Uniformity::None
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct InsertedSet(pub SparseBitSet);

impl Printable for InsertedSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Inserted({{{}}})", set)
    }
}

impl LatticeValue for InsertedSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type Inserted = DenseLattice<InsertedSet>;
pub type InsertedAnalysis = DenseForward<InsertedForward>;

#[derive(Default)]
pub struct InsertedForward;

impl InsertedForward {
    fn insert_in_preds(
        this: &InsertedAnalysis,
        ctx: &Context,
        solver: &DataflowSolver,
        point: ProgramPoint,
        predecessors: &[ReadRef<Inserted>],
    ) {
        if predecessors.len() <= 1 {
            return;
        }

        let predecessors = predecessors
            .iter()
            .map(|it| it.deref().anchor())
            .collect::<SmallVec<[_; 4]>>();
        let available_preds = predecessors
            .iter()
            .map(|&pred| solver.get_or_create_for::<InsertedAnalysis, Available>(point, pred))
            .collect::<SmallVec<[_; 4]>>();
        let available_preds = available_preds
            .iter()
            .map(|it| Ref::map(it.deref(), |it| it.value().0.unwrap()))
            .collect::<SmallVec<[_; 4]>>();

        let avail = solver.get_or_create_for::<InsertedAnalysis, Available>(point, point);
        let antic = solver.get_or_create_for::<InsertedAnalysis, Anticipated>(point, point);
        let available_anywhere: SparseBitSet =
            hi_sparse_bitset::reduce(ops::Or, available_preds.iter().map(|it| &**it))
                .unwrap()
                .intersection(antic.deref().value().0.unwrap())
                .difference(avail.deref().value().0.unwrap())
                .into();

        for (&pred, available) in predecessors.iter().zip(available_preds) {
            let antic_pred = solver.get_or_create_for::<InsertedAnalysis, Anticipated>(point, pred);
            let insert: SparseBitSet = available_anywhere
                .intersection(antic_pred.deref().value().0.unwrap())
                .difference(&*available)
                .into();

            let computes = solver.get_or_create_mut::<Computes>(pred);
            solver.update_state(ctx, &computes, |it| it.join(&ComputesSet(insert.clone())));

            let lattice = this.get_lattice_mut(solver, pred);
            solver.update_state(ctx, &lattice, |it| it.join(&InsertedSet(insert)));
        }
    }
}

impl DenseForwardDataflowAnalysis for InsertedForward {
    type LatticeValue = InsertedSet;

    fn visit_operation(
        _: &DenseForward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: Ptr<Operation>,
        _: &ReadRef<Inserted>,
        _: &WriteRef<Inserted>,
    ) -> Result<()> {
        Ok(())
    }

    fn visit_block_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        _: Ptr<BasicBlock>,
        point: ProgramPoint,
        predecessors: &[ReadRef<Inserted>],
        _: &WriteRef<Inserted>,
    ) {
        Self::insert_in_preds(this, ctx, solver, point, predecessors);
    }

    fn visit_region_branch_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        point: ProgramPoint,
        predecessors: &[ReadRef<Inserted>],
        _: &WriteRef<Inserted>,
    ) {
        if branch
            .get_operation()
            .impls::<dyn HoistableRegionOpInterface>(ctx)
        {
            Self::insert_in_preds(this, ctx, solver, point, predecessors);
        }
    }

    // Unhandled only
    fn visit_control_flow_transfer(
        _: &DenseForward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: &[ReadRef<Inserted>],
        _: &WriteRef<Inserted>,
    ) {
    }

    fn set_to_entry_state(
        _: &DenseForward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: &WriteRef<Inserted>,
    ) {
    }
}

/// `Computed` propagated backwards through single-successor single-predecessor points.
/// Used to hoist computations that are computed on any direct successor edges, i.e. in the `then`
/// or `else` branches, but not computations that are after the `if` entirely.
#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct ComputedOnImmediateEdge(pub SparseBitSet);

impl Printable for ComputedOnImmediateEdge {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "ComputedOnImmediateEdge({{{}}})", set)
    }
}

impl LatticeValue for ComputedOnImmediateEdge {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type SparseInsertedAnalysis = DenseBackward<InsertedBackward>;

// Approximation of sparse code motion
#[derive(Default)]
pub struct InsertedBackward;

impl InsertedBackward {
    fn insert_in_pred(
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<ComputedOnImmediateEdge>>],
        before: &WriteRef<DenseLattice<ComputedOnImmediateEdge>>,
    ) {
        // Hoist all expressions that are computed on any successors, are anticipated, and not available
        let avail = solver.get_or_create_for::<SparseInsertedAnalysis, Available>(point, point);
        let antic = solver.get_or_create_for::<SparseInsertedAnalysis, Anticipated>(point, point);
        let successors = successors
            .iter()
            .map(|it| Ref::map(it.deref(), |it| &it.value().0))
            .collect::<SmallVec<[_; 4]>>();
        let computed_any_succs =
            hi_sparse_bitset::reduce(ops::Or, successors.iter().map(Deref::deref)).unwrap();
        let new_insert: SparseBitSet = computed_any_succs
            .intersection(antic.deref().value().0.unwrap())
            .difference(avail.deref().value().0.unwrap())
            .into();

        let insert = solver.get_or_create_mut::<Inserted>(point);
        let computes = solver.get_or_create_mut::<Computes>(point);

        solver.update_state(ctx, &insert, |it| it.join(&InsertedSet(new_insert.clone())));
        solver.update_state(ctx, &computes, |it| {
            it.join(&ComputesSet(new_insert.clone()))
        });
        solver.update_state(ctx, before, |it| {
            it.meet(&ComputedOnImmediateEdge(new_insert))
        });
    }
}

impl DenseBackwardDataflowAnalysis for InsertedBackward {
    type LatticeValue = ComputedOnImmediateEdge;

    fn visit_operation(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let before_op = ProgramPoint::before_op(ctx, op);
        let after_op = ProgramPoint::after_op(ctx, op);
        let computes =
            solver.get_or_create_for::<SparseInsertedAnalysis, Computes>(before_op, after_op);
        let before_val = after
            .deref()
            .value()
            .0
            .union(&computes.deref().value().0)
            .into();
        solver.update_state(ctx, before, |it| {
            it.meet(&ComputedOnImmediateEdge(before_val))
        });
        Ok(())
    }

    fn visit_block_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        _: Ptr<BasicBlock>,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        if successors.is_empty() {
            return Ok(());
        };
        if successors.len() == 1 {
            let succ = &successors[0];
            let block = succ.deref().anchor().block().unwrap();
            if block.num_preds(ctx) == 1 {
                this.meet(solver, ctx, before, succ);
            }
            return Ok(());
        }

        Self::insert_in_pred(solver, ctx, point, successors, before);
        Ok(())
    }

    fn visit_region_branch_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        region_from: RegionPredecessor,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        if successors.is_empty() {
            return Ok(());
        };
        if successors.len() == 1 {
            let succ_point = successors[0].deref().anchor();
            let reg_successors = branch.successor_regions(ctx, region_from);
            let successor = reg_successors
                .iter()
                .find(|succ| match succ {
                    RegionSuccessor::Region(region)
                        if let Some(entry) = region.deref(ctx).get_entry_block() =>
                    {
                        succ_point == ProgramPoint::at_block_start(ctx, entry)
                    }
                    RegionSuccessor::Region(_) | RegionSuccessor::AfterOp => {
                        succ_point == ProgramPoint::after_op(ctx, branch.get_operation())
                    }
                })
                .expect("successor should be in successor list");
            if branch.predecessors(ctx, *successor).len() == 1 {
                this.meet(solver, ctx, before, &successors[0]);
            }
            return Ok(());
        }

        if branch
            .get_operation()
            .impls::<dyn HoistableRegionOpInterface>(ctx)
        {
            Self::insert_in_pred(solver, ctx, point, successors, before);
        }
        Ok(())
    }

    fn visit_control_flow_transfer(
        _: &DenseBackward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: ProgramPoint,
        _: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        _: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        // Unhandled control flow transfers nothing
        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
    }
}
