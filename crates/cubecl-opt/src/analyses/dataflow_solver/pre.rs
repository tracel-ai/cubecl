//! Partial Redundancy Elimination, using techniques from:
//! * [GVN-PRE; Vandrunen, Thomas & Hosking, Antony. (2004). Value-Based Partial Redundancy
//!   Elimination.](https://doi.org/10.1007/978-3-540-24723-4_12)
//! * LLVM's `MemorySSA`, adapted to the region-based IR and expanded with interprocedural support
//! * Speculative Code Motion, loosely based on
//!   [Fukuhara, Junji and Munehiro Takimoto. “Branch Divergence Reduction Based on Code Motion.” J.
//!   Inf. Process. 28 (2020): 302-309.](https://doi.org/10.2197/ipsjjip.28.302)
//!

use core::{cell::Ref, fmt};

use alloc::format;
use cubecl_ir::{
    dialect::OperationExt,
    interfaces::{
        control_flow::{RegionBranchOpInterface, RegionPredecessor, RegionSuccessor},
        memory_slot::MemoryValue,
        side_effects::{ConditionallySpeculatable, Speculatability},
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
    graph::walkers::uninterruptible::immutable::walk_op,
    opts::dce::SideEffects,
    printable::{self, Printable},
    utils::table::HMap,
};
use smallvec::SmallVec;

use crate::{
    BitSetExt, BitsetLatticeExt, SparseBitSet,
    analyses::{
        dataflow_solver::{
            DataflowAnalysis, ProgramPoint, SmallPtrVec, SolverConfig,
            dead_code::{CFGEdge, Executable, is_block_live},
            dense::{
                DenseBackwardDataflowAnalysis, DenseForwardDataflowAnalysis, DenseLattice,
                LatticeValue,
            },
            sccp::DummySparseConstantPropagation,
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

/// Lazy Code Motion, best for CPU
pub fn compute_lcm_analyses(ctx: &Context, op: Ptr<Operation>) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DummySparseConstantPropagation);
    solver.load(ComputesAnalysis);
    solver.load(ValueDependentsAnalysis::default());
    solver.initialize_and_run(ctx, op)?;

    let finished_analyses = solver.loaded_analyses().to_vec();
    let memory_kill_sets = collect_memory_kill_sets(ctx, &solver, op);

    let should_init =
        |analysis: &dyn DataflowAnalysis| !finished_analyses.contains(&analysis.type_id());

    solver.load(AvailableAnalysis::default());
    solver.load(Anticipated::new(memory_kill_sets));
    solver.initialize_filtered_and_run(ctx, op, should_init)?;

    Ok(solver)
}

/// Speculative Lazy Code Motion, best for GPU because it moves expressions out of divergent
/// branches if possible.
pub fn compute_slcm_analyses(ctx: &Context, op: Ptr<Operation>) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DummySparseConstantPropagation);
    solver.load(ComputesAnalysis);
    solver.load(ValueDependentsAnalysis::default());
    solver.load(DynamicUniformityAnalysis::default());
    solver.initialize_and_run(ctx, op)?;

    let finished_analyses = solver.loaded_analyses().to_vec();
    let memory_kill_sets = collect_memory_kill_sets(ctx, &solver, op);

    let should_init =
        |analysis: &dyn DataflowAnalysis| !finished_analyses.contains(&analysis.type_id());

    solver.load(AvailableAnalysis::default());
    solver.load(Anticipated::new_speculative(memory_kill_sets));
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
pub struct AvailableSet(SparseBitSet);

impl Printable for AvailableSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Available({{{}}})", set)
    }
}

impl LatticeValue for AvailableSet {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type AvailableExpressions = DenseLattice<AvailableSet>;
pub type AvailableAnalysis = DenseForward<Available>;

#[derive(Default)]
pub struct Available;

impl DenseForwardDataflowAnalysis for Available {
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
        let computes = solver.get_or_create_for::<Self, Computes>(point, point);

        if after == before {
            Self::set_to_entry_state(this, solver, ctx, after);
            return Ok(());
        }

        let before = before.deref();
        let union = before.value().0.union(&**computes.deref().value()).into();
        solver.update_state(ctx, after, |it| it.join(&AvailableSet(union)));
        Ok(())
    }

    fn visit_control_flow_transfer(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        before: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let before: SmallVec<[_; 4]> = before.iter().map(|it| it.deref()).collect();
        let new_value = hi_sparse_bitset::reduce(ops::And, before.iter().map(|it| &it.value().0))
            .map(Into::into)
            .unwrap_or_default();
        solver.update_state(ctx, after, |it| it.join(&AvailableSet(new_value)));
    }

    fn set_to_entry_state(
        _: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.join(&Default::default()));
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

fn op_transitive_kill_set(
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
pub struct AnticipatedSet(SparseBitSet);

impl Printable for AnticipatedSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Anticipated({{{}}})", set)
    }
}

impl LatticeValue for AnticipatedSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type AnticipatedExpressions = DenseLattice<AnticipatedSet>;
pub type AnticipatedAnalysis = DenseBackward<Anticipated>;

pub struct Anticipated {
    memory_kill_sets: HMap<MemoryValue, SparseBitSet>,
    /// Use speculative code motion for divergent branches. Should be off on CPU and on on GPU.
    speculative: bool,
}

impl Anticipated {
    pub fn new(memory_kill_sets: HMap<MemoryValue, SparseBitSet>) -> AnticipatedAnalysis {
        DenseBackward::new(Self {
            memory_kill_sets,
            speculative: false,
        })
    }

    pub fn new_speculative(
        memory_kill_sets: HMap<MemoryValue, SparseBitSet>,
    ) -> AnticipatedAnalysis {
        DenseBackward::new(Self {
            memory_kill_sets,
            speculative: true,
        })
    }
}

impl DenseBackwardDataflowAnalysis for Anticipated {
    type LatticeValue = AnticipatedSet;

    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(ComputesAnalysis::new);
        if self.speculative {
            solver.ensure_loaded(SpeculatableAnalysis::default);
        }
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
        let point = ProgramPoint::before_op(ctx, op);
        let after_point = ProgramPoint::after_op(ctx, op);
        let computes = solver.get_or_create_for::<Self, Computes>(point, after_point);
        let computes = computes.deref();
        let before_set = after.value().0.union(&**computes.value());

        let before_set = if let Some(def) = memory_ssa.op_memory_def(op)
            && let Some(kill_set) = this.memory_kill_sets.get(&def)
        {
            before_set.difference(kill_set).into()
        } else {
            before_set.into()
        };
        drop(after);

        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(before_set)));
        Ok(())
    }

    fn visit_block(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let memory_ssa = analyses.get_analysis::<MemorySSA>(solver.root().unwrap(), ctx)?;

        let point = ProgramPoint::at_block_end(ctx, block);
        if !is_block_live::<Self>(solver, ctx, point) {
            return Ok(());
        }

        let before = this.get_lattice_mut(solver, point);

        if this.visit_block_special_cases(solver, ctx, block)? {
            return Ok(());
        };

        let mut after_sets = SmallVec::<[_; 4]>::new();
        let mut after_sets_speculative = SmallVec::<[_; 4]>::new();
        let is_divergent = this.speculative && is_terminator_divergent(ctx, solver, block);

        let speculatable = match is_divergent {
            true => {
                let speculatable =
                    solver.get_or_create_for::<AnticipatedAnalysis, Speculatable>(point, point);
                speculatable.deref().value().0.clone()
            }
            false => Default::default(),
        };

        for successor in block.deref(ctx).succs(ctx) {
            let edge = CFGEdge::new(block, successor).into();
            let executable = solver.get_or_create_for::<Self, Executable>(point, edge);
            if !executable.deref().is_live() {
                continue;
            }

            let successor_point = ProgramPoint::at_block_start(ctx, successor);
            if !is_block_live::<Self>(solver, ctx, successor_point) {
                continue;
            }

            let after = this.get_lattice_for(solver, point, successor_point);
            let after_set = after.deref().value().0.clone();
            let after_spec: SparseBitSet = if is_divergent {
                after_set.intersection(&speculatable).into()
            } else {
                Default::default()
            };

            if let Some(def) = memory_ssa.block_memory_def(successor)
                && let Some(kill_set) = this.memory_kill_sets.get(&def)
            {
                after_sets.push(after_set.difference(kill_set).into());
                after_sets_speculative.push(after_spec.difference(kill_set).into());
            } else {
                after_sets.push(after_set);
                after_sets_speculative.push(after_spec);
            }
        }

        let mut anticipated = hi_sparse_bitset::reduce(ops::And, after_sets.iter())
            .map(Into::into)
            .unwrap_or_default();

        if is_divergent {
            anticipated = hi_sparse_bitset::reduce(ops::Or, after_sets_speculative.iter())
                .map(|it| it.union(&anticipated).into())
                .unwrap_or_default();
        }
        solver.update_state(ctx, &before, |it| it.meet(&AnticipatedSet(anticipated)));

        Ok(())
    }

    fn visit_region_branch_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        predecessor: RegionPredecessor,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let memory_ssa = analyses.get_analysis::<MemorySSA>(solver.root().unwrap(), ctx)?;

        let branch_op = branch.get_operation();
        let successors = branch.successor_regions(ctx, predecessor);

        let mut after_sets = SmallVec::<[_; 4]>::new();
        let mut after_sets_speculative = SmallVec::<[_; 4]>::new();
        let is_divergent =
            this.speculative && is_region_op_divergent(ctx, solver, branch_op, predecessor);

        let speculatable = match is_divergent {
            true => {
                let speculatable =
                    solver.get_or_create_for::<AnticipatedAnalysis, Speculatable>(point, point);
                speculatable.deref().value().0.clone()
            }
            false => Default::default(),
        };

        for successor in successors {
            let (def, after, after_spec) = match successor {
                RegionSuccessor::AfterOp => {
                    let after_point = ProgramPoint::after_op(ctx, branch_op);
                    let after = this.get_lattice_for(solver, point, after_point);
                    let after = after.deref().value().0.clone();
                    // Not relevant for any current region ops, maybe though?
                    let after_spec: SparseBitSet = Default::default();
                    (memory_ssa.op_memory_def(branch_op), after, after_spec)
                }
                RegionSuccessor::Region(successor_region)
                    if let Some(entry) = successor_region.deref(ctx).get_entry_block() =>
                {
                    let successor_point = ProgramPoint::at_block_start(ctx, entry);
                    if !is_block_live::<Self>(solver, ctx, successor_point) {
                        continue;
                    }

                    let after = this.get_lattice_for(solver, point, successor_point);
                    let after_set = after.deref().value().0.clone();
                    let after_spec = if is_divergent {
                        after_set.intersection(&speculatable).into()
                    } else {
                        Default::default()
                    };

                    (memory_ssa.block_memory_def(entry), after_set, after_spec)
                }
                RegionSuccessor::Region(_) => {
                    let after_point = ProgramPoint::after_op(ctx, branch_op);
                    let after = this.get_lattice_for(solver, point, after_point);
                    let after = after.deref().value().0.clone();
                    (None, after, Default::default())
                }
            };

            if let Some(def) = def
                && let Some(kill_set) = this.memory_kill_sets.get(&def)
            {
                after_sets.push(after.difference(kill_set).into());
                after_sets_speculative.push(after_spec.difference(kill_set).into());
            } else {
                after_sets.push(after);
                after_sets_speculative.push(after_spec);
            }
        }

        let mut anticipated = hi_sparse_bitset::reduce(ops::And, after_sets.iter())
            .map(Into::into)
            .unwrap_or_default();

        if is_divergent {
            anticipated = hi_sparse_bitset::reduce(ops::Or, after_sets_speculative.iter())
                .map(|it| it.union(&anticipated).into())
                .unwrap_or_default();
        }
        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(anticipated)));

        Ok(())
    }

    fn visit_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        succs: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let mut analyses = solver.analyses();
        let memory_ssa = analyses.get_analysis::<MemorySSA>(solver.root().unwrap(), ctx)?;

        let mut after_sets = SmallVec::<[SparseBitSet; 4]>::new();
        for succ in succs {
            let point = succ.deref().anchor();
            let def = match point.prev_op(ctx) {
                Some(op) => memory_ssa.op_memory_def(op),
                None => memory_ssa.block_memory_def(point.block().unwrap()),
            };
            let Some(def) = def else {
                after_sets.push(succ.deref().value().0.clone());
                continue;
            };
            let Some(kill_set) = this.memory_kill_sets.get(&def) else {
                after_sets.push(succ.deref().value().0.clone());
                continue;
            };

            after_sets.push(succ.deref().value().difference(kill_set).into());
        }

        let new_value = hi_sparse_bitset::reduce(ops::And, after_sets.iter())
            .map(Into::into)
            .unwrap_or_default();
        solver.update_state(ctx, before, |it| it.meet(&AnticipatedSet(new_value)));

        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.meet(&Default::default()));
    }
}

fn is_terminator_divergent(ctx: &Context, solver: &DataflowSolver, block: Ptr<BasicBlock>) -> bool {
    let Some(term) = block.deref(ctx).get_terminator(ctx) else {
        return false;
    };
    let Some(branch_op) = term.cast::<dyn UniformBranchOpInterface>(ctx) else {
        return false;
    };
    let mut operands = SmallPtrVec::new();
    for opd in term.deref(ctx).operands() {
        let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
            panic!("Dynamic uniformity should be converged before anticipated sets are run")
        };
        operands.push(lattice.deref().value().0);
    }
    branch_op.successor_uniformity(ctx, &operands) <= Uniformity::None
}

fn is_region_op_divergent(
    ctx: &Context,
    solver: &DataflowSolver,
    op: Ptr<Operation>,
    from: RegionPredecessor,
) -> bool {
    let Some(region_op) = op.cast::<dyn UniformRegionOpInterface>(ctx) else {
        return false;
    };
    match from {
        RegionPredecessor::Parent => {
            let mut operands = SmallPtrVec::new();
            for opd in op.deref(ctx).operands() {
                let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
                    panic!("Dynamic uniformity should be converged before anticipated sets are run")
                };
                operands.push(lattice.deref().value().0);
            }
            region_op.entry_successor_region_uniformity(ctx, &operands) <= Uniformity::None
        }
        RegionPredecessor::Terminator(term)
            if let Some(uniform_term) = term
                .get_operation()
                .cast::<dyn UniformRegionTerminatorOpInterface>(ctx) =>
        {
            let mut operands = SmallPtrVec::new();
            for opd in op.deref(ctx).operands() {
                let Some(lattice) = solver.lookup_state::<DynamicUniformityLattice>(opd) else {
                    panic!("Dynamic uniformity should be converged before anticipated sets are run")
                };
                operands.push(lattice.deref().value().0);
            }
            uniform_term.successor_region_uniformity(ctx, &operands) <= Uniformity::None
        }
        RegionPredecessor::Terminator(_) => false,
    }
}

#[derive(Deref, DerefMut, PartialEq, Eq, Default)]
pub struct SpeculatableSet(SparseBitSet);

impl Printable for SpeculatableSet {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set = self.0.iter().map(|i| format!("e{i}")).join(", ");
        write!(f, "Speculatable({{{}}})", set)
    }
}

impl LatticeValue for SpeculatableSet {
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        self.unite_changed(rhs)
    }
}

pub type SpeculatableAnalysis = DenseBackward<SpeculatableDFA>;
pub type Speculatable = DenseLattice<SpeculatableSet>;

#[derive(Default)]
pub struct SpeculatableDFA;

impl DenseBackwardDataflowAnalysis for SpeculatableDFA {
    type LatticeValue = SpeculatableSet;

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let before_point = ProgramPoint::before_op(ctx, op);
        let after_point = ProgramPoint::after_op(ctx, op);
        let computes =
            solver.get_or_create_for::<DenseBackward<Self>, Computes>(before_point, after_point);

        if after == before {
            Self::set_to_exit_state(this, solver, ctx, before);
            return Ok(());
        }

        let after = after.deref();
        let after = &after.value().0;

        let has_side_effects = op
            .cast::<dyn SideEffects>(ctx)
            .is_none_or(|op| op.has_side_effects(ctx));
        let is_speculatable = !has_side_effects && is_speculatable(ctx, op);

        let before_set = if is_speculatable {
            after.union(&computes.deref().value().0).into()
        } else {
            let dependents = op_transitive_kill_set(solver, &computes, op.deref(ctx).results());
            after
                .difference(&computes.deref().value().0)
                .difference(&dependents)
                .into()
        };
        solver.update_state(ctx, before, |before| {
            before.meet(&SpeculatableSet(before_set))
        });
        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.meet(&Default::default()));
    }
}

fn is_speculatable(ctx: &Context, op: Ptr<Operation>) -> bool {
    let speculatable = match op.cast::<dyn ConditionallySpeculatable>(ctx) {
        Some(op) => op.speculatability(ctx),
        None => Speculatability::NotSpeculatable,
    };
    match speculatable {
        Speculatability::NotSpeculatable => false,
        Speculatability::Speculatable => true,
        Speculatability::RecursivelySpeculatable => is_recursively_speculatable(ctx, op),
    }
}

fn is_recursively_speculatable(ctx: &Context, op: Ptr<Operation>) -> bool {
    for op in op.deref(ctx).immediately_nested_ops(ctx) {
        if !is_speculatable(ctx, op) {
            return false;
        }
    }
    true
}
