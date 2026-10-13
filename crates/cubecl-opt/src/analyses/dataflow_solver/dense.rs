use core::{any::TypeId, cell::RefCell};

use cubecl_ir::{
    dialect::{BlockPtrExt, RegionPtrExt},
    interfaces::control_flow::{
        CallableOpInterface, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
        RegionPredecessor, RegionSuccessor,
    },
    prelude::*,
};
use derive_more::Deref;
use pliron::{
    basic_block::BasicBlock, graph::walkers::uninterruptible::immutable::walk_op,
    linked_list::ContainsLinkedList, printable::Printable, symbol_table::SymbolTableCollection,
    utils::table::ISet,
};
use smallvec::SmallVec;

use crate::analyses::dataflow_solver::{
    AnalysisState, ChangeResult, DataflowAnalysis, ReadRef, SolverWorkItem, WriteRef,
    dead_code::{CFGEdge, DeadCodeAnalysis, Executable, PredecessorState, is_block_live},
};

use super::{DataflowSolver, ProgramPoint};

pub trait LatticeValue: Default + PartialEq + Printable + Sized + 'static {
    fn join(&mut self, rhs: &Self) -> ChangeResult {
        let _ = rhs;
        ChangeResult::Unchanged
    }
    fn meet(&mut self, rhs: &Self) -> ChangeResult {
        let _ = rhs;
        ChangeResult::Unchanged
    }
}

pub struct DenseLattice<T: LatticeValue> {
    anchor: ProgramPoint,
    pub value: T,
    dependents: RefCell<ISet<SolverWorkItem>>,
}

impl<T: LatticeValue> Printable for DenseLattice<T> {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}: {}", self.anchor.disp(ctx), self.value.disp(ctx))
    }
}

impl<T: LatticeValue> DenseLattice<T> {
    pub fn join(&mut self, rhs: &T) -> ChangeResult {
        self.value.join(rhs)
    }

    pub fn meet(&mut self, rhs: &T) -> ChangeResult {
        self.value.meet(rhs)
    }

    pub fn anchor(&self) -> ProgramPoint {
        self.anchor
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: LatticeValue> AnalysisState for DenseLattice<T> {
    type Anchor = ProgramPoint;

    fn create(anchor: ProgramPoint) -> Self {
        Self {
            anchor,
            value: Default::default(),
            dependents: Default::default(),
        }
    }

    fn add_dependency<A: 'static>(&self, point: ProgramPoint) {
        self.dependents
            .borrow_mut()
            .insert((point, TypeId::of::<A>()));
    }

    fn on_update(&self, _ctx: &Context, solver: &DataflowSolver) {
        for dependent in self.dependents.borrow().iter() {
            solver.enqueue(*dependent);
        }
    }
}

pub trait DenseForwardDataflowAnalysis: Sized + 'static {
    type LatticeValue: LatticeValue;

    fn on_load(&self, solver: &mut DataflowSolver) {
        let _ = solver;
    }

    /// Verify analysis can be run on the solver. Should be used to verify required analyses are
    /// loaded.
    fn verify(solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        let _ = (solver, ctx, root);
        Ok(())
    }

    fn visit_operation(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()>;

    fn set_to_entry_state(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    );

    fn build_operation_equivalent_lattice_anchor(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) {
        let _ = (this, solver, ctx, op);
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_block_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        point: ProgramPoint,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let _ = (block, point);
        Self::visit_control_flow_transfer(this, solver, ctx, predecessors, after);
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_region_branch_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        point: ProgramPoint,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let _ = (branch, point);
        Self::visit_control_flow_transfer(this, solver, ctx, predecessors, after);
    }

    fn visit_callee_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        Self::visit_control_flow_transfer(this, solver, ctx, predecessors, after);
    }

    fn visit_return_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        Self::visit_control_flow_transfer(this, solver, ctx, predecessors, after);
    }

    fn visit_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        predecessors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        for predecessor in predecessors {
            this.join(solver, ctx, after, predecessor);
        }
    }

    fn visit_external_call(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        let _ = (call, before);
        Self::set_to_entry_state(this, solver, ctx, after);
    }
}

#[derive(Default, Deref)]
pub struct DenseForward<T: DenseForwardDataflowAnalysis> {
    #[deref]
    inner: T,
    symbol_table: RefCell<SymbolTableCollection>,
}

impl<T: DenseForwardDataflowAnalysis> DenseForward<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            symbol_table: Default::default(),
        }
    }
}

impl<T: DenseForwardDataflowAnalysis> DataflowAnalysis for DenseForward<T> {
    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(DeadCodeAnalysis::default);
        self.inner.on_load(solver);
    }

    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<DeadCodeAnalysis>()?;
        T::verify(solver, ctx, root)
    }

    fn initialize(
        &mut self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        self.process_operation(solver, ctx, root)?;

        for region in root.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                self.visit_block(solver, ctx, block);
                for op in block.deref(ctx).iter(ctx) {
                    self.initialize(solver, ctx, op)?;
                }
            }
        }
        Ok(())
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if let Some(op) = point.prev_op(ctx) {
            return self.process_operation(solver, ctx, op);
        }
        self.visit_block(solver, ctx, point.block().unwrap());
        Ok(())
    }

    fn initialize_equivalent_lattice_anchor(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) {
        walk_op(
            ctx,
            &mut (self, solver),
            &WALKCONFIG_ANY,
            root,
            |ctx, (this, solver), node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                if op.impls::<dyn RegionBranchOpInterface>(ctx)
                    || op.impls::<dyn CallOpInterface>(ctx)
                {
                    return;
                }
                T::build_operation_equivalent_lattice_anchor(this, solver, ctx, op);
            },
        );
    }
}

impl<T: DenseForwardDataflowAnalysis> DenseForward<T> {
    pub fn get_lattice<'a>(
        &self,
        solver: &'a DataflowSolver,
        anchor: ProgramPoint,
    ) -> ReadRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create(anchor)
    }

    pub fn get_lattice_mut<'a>(
        &self,
        solver: &'a DataflowSolver,
        anchor: ProgramPoint,
    ) -> WriteRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create_mut(anchor)
    }

    pub fn get_lattice_for<'a>(
        &self,
        solver: &'a DataflowSolver,
        point: ProgramPoint,
        anchor: ProgramPoint,
    ) -> ReadRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create_for::<Self, DenseLattice<T::LatticeValue>>(point, anchor)
    }

    pub fn join(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        lhs: &WriteRef<DenseLattice<T::LatticeValue>>,
        rhs: &ReadRef<DenseLattice<T::LatticeValue>>,
    ) {
        if lhs == rhs {
            return;
        }
        solver.update_state(ctx, lhs, |lhs| lhs.join(rhs.deref().value()));
    }

    pub fn process_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let point = ProgramPoint::after_op(ctx, op);
        if !is_block_live::<Self>(solver, ctx, point) {
            return Ok(());
        }

        // Get the dense lattice to update.
        let after = self.get_lattice_mut(solver, point);

        // Get the dense state before the execution of the op.
        let before = self.get_lattice_for(solver, point, ProgramPoint::before_op(ctx, op));

        // If this op implements region control-flow, then control-flow dictates its
        // transfer function.
        if let Some(branch) = op.cast::<dyn RegionBranchOpInterface>(ctx) {
            self.visit_region_branch_operation(solver, ctx, point, &*branch, after);
            return Ok(());
        }

        // If this is a call operation, then join its lattices across known return
        // sites.
        if let Some(call) = op.cast::<dyn CallOpInterface>(ctx) {
            self.visit_call_operation(solver, ctx, &*call, &before, &after);
            return Ok(());
        }

        T::visit_operation(self, solver, ctx, op, &before, &after)
    }

    pub fn visit_block(&self, solver: &DataflowSolver, ctx: &Context, block: Ptr<BasicBlock>) {
        let point = ProgramPoint::at_block_start(ctx, block);
        if !is_block_live::<Self>(solver, ctx, point) {
            return;
        }

        // Get the dense lattice to update.
        let after = self.get_lattice_mut(solver, point);

        // The dense lattices of entry blocks are set by region control-flow or the
        // callgraph.
        if block.is_entry_block(ctx) {
            let parent_region = block.deref(ctx).get_parent_region().unwrap();
            let parent_op = block.deref(ctx).get_parent_op(ctx).unwrap();
            if let Some(callable) = parent_op.cast::<dyn CallableOpInterface>(ctx)
                && callable.callable_region(ctx) == Some(parent_region)
            {
                let callsites = solver.get_or_create_for::<Self, PredecessorState>(
                    point,
                    ProgramPoint::after_op(ctx, parent_op),
                );
                // If not all callsites are known, conservatively mark all lattices as
                // having reached their pessimistic fixpoints. Do the same if
                // interprocedural analysis is not enabled.
                if !callsites.deref().all_predecessors_known()
                    || !solver.config().is_interprocedural
                {
                    return T::set_to_entry_state(self, solver, ctx, &after);
                }

                let mut predecessors = SmallVec::<[_; 4]>::new();
                for callsite in callsites.deref().known_predecessors() {
                    // Get the dense lattice before the callsite.
                    let before = self.get_lattice_for(
                        solver,
                        point,
                        ProgramPoint::before_op(ctx, *callsite),
                    );

                    predecessors.push(before);
                }
                T::visit_callee_control_flow_transfer(self, solver, ctx, &predecessors, &after);
                return;
            }

            if let Some(branch) = parent_op.cast::<dyn RegionBranchOpInterface>(ctx) {
                return self.visit_region_branch_operation(solver, ctx, point, &*branch, after);
            }

            return T::set_to_entry_state(self, solver, ctx, &after);
        }

        // Join the state with the state after the block's predecessors.
        let mut predecessors = SmallVec::<[_; 4]>::new();
        for predecessor in block.preds(ctx) {
            // Skip control edges that aren't executable.
            let executable = solver.get_or_create_for::<Self, Executable>(
                point,
                CFGEdge::new(predecessor, block).into(),
            );
            if !executable.deref().is_live() {
                continue;
            }

            let term = predecessor.deref(ctx).get_terminator(ctx).unwrap();
            let before = self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, term));
            predecessors.push(before);
        }

        // Merge in the state from the predecessors' terminators.
        T::visit_block_transfer(self, solver, ctx, block, point, &predecessors, &after);
    }

    pub fn visit_call_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        before: &ReadRef<DenseLattice<T::LatticeValue>>,
        after: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        let call_op = call.get_operation();
        let is_external_callable = || {
            let Some(callable) = self.resolve_callable(ctx, call) else {
                return false;
            };
            let callable = callable.cast::<dyn CallableOpInterface>(ctx);
            callable.is_some_and(|callable| callable.callable_region(ctx).is_none())
        };
        if !solver.config().is_interprocedural || is_external_callable() {
            return T::visit_external_call(self, solver, ctx, call, before, after);
        }

        let predecessors = solver.get_or_create_for::<Self, PredecessorState>(
            ProgramPoint::after_op(ctx, call_op),
            ProgramPoint::after_op(ctx, call_op),
        );
        if !predecessors.deref().all_predecessors_known() {
            return T::set_to_entry_state(self, solver, ctx, after);
        }

        let mut predecessor_lattices = SmallVec::<[_; 4]>::new();
        for &predecessor in predecessors.deref().known_predecessors() {
            // Get the lattices at callee return:
            //
            //   builtin.func @callee() {
            //     ...
            //     return  // predecessor
            //     // latticeAtCalleeReturn
            //   }
            //   builtin.func @caller() {
            //     ...
            //     call @callee
            //     // latticeAfterCall
            //     ...
            //   }
            let lattice_at_callee_return = self.get_lattice_for(
                solver,
                ProgramPoint::after_op(ctx, call_op),
                ProgramPoint::after_op(ctx, predecessor),
            );
            predecessor_lattices.push(lattice_at_callee_return);
        }

        T::visit_return_control_flow_transfer(self, solver, ctx, &predecessor_lattices, after);
    }

    pub fn visit_region_branch_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        after: WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        let branch_op = branch.get_operation();
        let predecessors = solver.get_or_create_for::<Self, PredecessorState>(point, point);
        assert!(
            predecessors.deref().all_predecessors_known(),
            "unexpected unresolved region successors"
        );

        let mut predecessor_lattices = SmallVec::<[_; 4]>::new();
        for &op in predecessors.deref().known_predecessors() {
            let pred_point = match op == branch_op {
                true => ProgramPoint::before_op(ctx, op),
                false => ProgramPoint::after_op(ctx, op),
            };
            let before = self.get_lattice_for(solver, point, pred_point);

            // This function is called in two cases:
            //   1. when visiting the block (point = block start);
            //   2. when visiting the parent operation (point = iter after parent op).
            // In both cases, we are looking for predecessor operations of the point,
            //   1. predecessor may be the terminator of another block from another
            //   region (assuming that the block does belong to another region via an
            //   assertion) or the parent (when parent can transfer control to this
            //   region);
            //   2. predecessor may be the terminator of a block that exits the
            //   region (when region transfers control to the parent) or the operation
            //   before the parent.
            // In the latter case, just perform the join as it isn't the control flow
            // affected by the region.
            if point.is_block_start(ctx) {
                predecessor_lattices.push(before);
            } else {
                let parent_op = op.deref(ctx).get_parent_op(ctx).unwrap();
                if parent_op == branch_op || op == branch_op {
                    predecessor_lattices.push(before);
                } else {
                    self.join(solver, ctx, &after, &before);
                }
            }
        }

        T::visit_region_branch_control_flow_transfer(
            self,
            solver,
            ctx,
            branch,
            point,
            &predecessor_lattices,
            &after,
        );
    }

    fn resolve_callable(
        &self,
        ctx: &Context,
        call: &dyn CallOpInterface,
    ) -> Option<Ptr<Operation>> {
        match call.callee(ctx) {
            CallOpCallable::Direct(symbol) => self
                .symbol_table
                .borrow_mut()
                .lookup_symbol_in_nearest_table(ctx, call.get_operation(), &symbol)
                .map(|it| it.get_operation()),
            CallOpCallable::Indirect(value) => value.defining_op(),
        }
    }
}

pub trait DenseBackwardDataflowAnalysis: Sized + 'static {
    type LatticeValue: LatticeValue;

    fn on_load(&self, solver: &mut DataflowSolver) {
        let _ = solver;
    }

    /// Verify analysis can be run on the solver. Should be used to verify required analyses are
    /// loaded.
    fn verify(solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        let _ = (solver, ctx, root);
        Ok(())
    }

    fn build_operation_equivalent_lattice_anchor(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) {
        let _ = (this, solver, ctx, op);
    }

    fn visit_operation(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()>;

    fn set_to_exit_state(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<DenseLattice<Self::LatticeValue>>,
    );

    fn visit_block(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
    ) -> Result<()> {
        this.visit_block(solver, ctx, block)
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_block_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let _ = (block, point);
        Self::visit_control_flow_transfer(this, solver, ctx, point, successors, before)
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
        this.visit_region_branch_operation(solver, ctx, point, branch, predecessor, before)
    }

    #[allow(clippy::too_many_arguments)]
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
        let _ = (branch, region_from, point);
        Self::visit_control_flow_transfer(this, solver, ctx, point, successors, before)
    }

    fn visit_call_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let _ = call;
        Self::visit_control_flow_transfer(
            this,
            solver,
            ctx,
            ProgramPoint::before_op(ctx, call.get_operation()),
            core::slice::from_ref(after),
            before,
        )
    }

    fn visit_return_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        Self::visit_control_flow_transfer(this, solver, ctx, point, successors, before)
    }

    fn visit_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        successors: &[ReadRef<DenseLattice<Self::LatticeValue>>],
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let _ = point;
        for successor in successors {
            this.meet(solver, ctx, before, successor);
        }
        Ok(())
    }

    fn visit_external_call(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) -> Result<()> {
        let _ = (call, after);
        Self::set_to_exit_state(this, solver, ctx, before);
        Ok(())
    }
}

#[derive(Default, Deref)]
pub struct DenseBackward<T: DenseBackwardDataflowAnalysis> {
    #[deref]
    inner: T,
    symbol_table: RefCell<SymbolTableCollection>,
}

impl<T: DenseBackwardDataflowAnalysis> DenseBackward<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            symbol_table: Default::default(),
        }
    }
}

impl<T: DenseBackwardDataflowAnalysis> DataflowAnalysis for DenseBackward<T> {
    fn on_load(&self, solver: &mut DataflowSolver) {
        solver.ensure_loaded(DeadCodeAnalysis::default);
        self.inner.on_load(solver);
    }

    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        T::verify(solver, ctx, root)
    }

    fn initialize(
        &mut self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        self.process_operation(solver, ctx, root)?;

        for region in root.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                self.visit_block(solver, ctx, block)?;
                for op in block.deref(ctx).iter(ctx).rev() {
                    self.initialize(solver, ctx, op)?;
                }
            }
        }
        Ok(())
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if let Some(op) = point.next_op(ctx) {
            return self.process_operation(solver, ctx, op);
        }
        self.visit_block(solver, ctx, point.block().unwrap())?;
        Ok(())
    }

    fn initialize_equivalent_lattice_anchor(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) {
        walk_op(
            ctx,
            &mut (self, solver),
            &WALKCONFIG_ANY,
            root,
            |ctx, (this, solver), node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                if op.impls::<dyn RegionBranchOpInterface>(ctx)
                    || op.impls::<dyn CallOpInterface>(ctx)
                {
                    return;
                }
                T::build_operation_equivalent_lattice_anchor(this, solver, ctx, op);
            },
        );
    }
}

impl<T: DenseBackwardDataflowAnalysis> DenseBackward<T> {
    pub fn get_lattice<'a>(
        &self,
        solver: &'a DataflowSolver,
        anchor: ProgramPoint,
    ) -> ReadRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create(anchor)
    }

    pub fn get_lattice_mut<'a>(
        &self,
        solver: &'a DataflowSolver,
        anchor: ProgramPoint,
    ) -> WriteRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create_mut(anchor)
    }

    pub fn get_lattice_for<'a>(
        &self,
        solver: &'a DataflowSolver,
        point: ProgramPoint,
        anchor: ProgramPoint,
    ) -> ReadRef<'a, DenseLattice<T::LatticeValue>> {
        solver.get_or_create_for::<Self, DenseLattice<T::LatticeValue>>(point, anchor)
    }

    pub fn meet(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        lhs: &WriteRef<DenseLattice<T::LatticeValue>>,
        rhs: &ReadRef<DenseLattice<T::LatticeValue>>,
    ) {
        if lhs == rhs {
            return;
        }
        solver.update_state(ctx, lhs, |lhs| lhs.meet(rhs.deref().value()));
    }

    fn process_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let point = ProgramPoint::before_op(ctx, op);
        if !is_block_live::<Self>(solver, ctx, point) {
            return Ok(());
        }

        let before = self.get_lattice_mut(solver, point);
        let after = self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, op));

        if let Some(branch) = op.cast::<dyn RegionBranchOpInterface>(ctx) {
            return T::visit_region_branch_operation(
                self,
                solver,
                ctx,
                point,
                &*branch,
                RegionPredecessor::Parent,
                &before,
            );
        }
        if let Some(call) = op.cast::<dyn CallOpInterface>(ctx) {
            return self.visit_call_operation(solver, ctx, &*call, &after, &before);
        }

        T::visit_operation(self, solver, ctx, op, &after, &before)
    }

    pub fn visit_block(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
    ) -> Result<()> {
        let point = ProgramPoint::at_block_end(ctx, block);
        if !is_block_live::<Self>(solver, ctx, point) {
            return Ok(());
        }

        let before = self.get_lattice_mut(solver, point);

        if self.visit_block_special_cases(solver, ctx, block)? {
            return Ok(());
        }

        let mut successors = SmallVec::<[_; 4]>::new();
        for successor in block.deref(ctx).succs(ctx) {
            let edge = CFGEdge::new(block, successor).into();
            let executable = solver.get_or_create_for::<Self, Executable>(point, edge);
            if !executable.deref().is_live() {
                continue;
            }

            let after =
                self.get_lattice_for(solver, point, ProgramPoint::at_block_start(ctx, successor));
            successors.push(after);
        }
        T::visit_block_transfer(self, solver, ctx, block, point, &successors, &before)?;
        Ok(())
    }

    /// Terrible name, need to come up with something better
    pub fn visit_block_special_cases(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
    ) -> Result<bool> {
        let parent_region = block.deref(ctx).get_parent_region().unwrap();
        let parent_op = block.deref(ctx).get_parent_op(ctx).unwrap();

        let point = ProgramPoint::at_block_end(ctx, block);
        if !is_block_live::<Self>(solver, ctx, point) {
            return Ok(true);
        }

        let before = self.get_lattice_mut(solver, point);

        let is_exit_block = |block: Ptr<BasicBlock>| {
            let Some(term) = block.deref(ctx).get_terminator(ctx) else {
                return true;
            };
            if block.is_empty(ctx) || !term.impls::<dyn IsTerminatorInterface>(ctx) {
                return true;
            }
            term.impls::<dyn RegionBranchTerminatorOpInterface>(ctx)
        };
        if is_exit_block(block) {
            // If this block is exiting from a callable, the successors of exiting from
            // a callable are the successors of all call sites. And the call sites
            // themselves are predecessors of the callable.

            if let Some(callable) = parent_op.cast::<dyn CallableOpInterface>(ctx)
                && callable.callable_region(ctx) == Some(parent_region)
            {
                let callsites = solver.get_or_create_for::<Self, PredecessorState>(
                    point,
                    ProgramPoint::after_op(ctx, parent_op),
                );
                if !callsites.deref().all_predecessors_known()
                    || !solver.config().is_interprocedural
                {
                    T::set_to_exit_state(self, solver, ctx, &before);
                    return Ok(true);
                }

                let mut successors = SmallVec::<[_; 4]>::new();
                for &callsite in callsites.deref().known_predecessors() {
                    let after =
                        self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, callsite));
                    successors.push(after);
                }
                T::visit_return_control_flow_transfer(
                    self,
                    solver,
                    ctx,
                    point,
                    &successors,
                    &before,
                )?;
                return Ok(true);
            }

            if let Some(branch) = parent_op.cast::<dyn RegionBranchOpInterface>(ctx) {
                let term = block.deref(ctx).get_terminator(ctx).unwrap();
                let terminator = TraitOp::try_from_op(term, ctx).unwrap();
                T::visit_region_branch_operation(
                    self,
                    solver,
                    ctx,
                    point,
                    &*branch,
                    RegionPredecessor::Terminator(terminator),
                    &before,
                )?;
                return Ok(true);
            }

            T::set_to_exit_state(self, solver, ctx, &before);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn visit_region_branch_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        predecessor: RegionPredecessor,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) -> Result<()> {
        let branch_op = branch.get_operation();
        let successors = branch.successor_regions(ctx, predecessor);

        let mut successor_lattices = SmallVec::<[_; 4]>::new();
        for successor in successors {
            let after = match successor {
                RegionSuccessor::AfterOp => {
                    self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, branch_op))
                }
                RegionSuccessor::Region(region) if region.is_empty(ctx) => {
                    self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, branch_op))
                }
                RegionSuccessor::Region(successor_region) => {
                    let successor_block = successor_region.deref(ctx).get_entry_block().unwrap();
                    let successor_point = ProgramPoint::at_block_start(ctx, successor_block);
                    if !is_block_live::<Self>(solver, ctx, successor_point) {
                        continue;
                    }
                    self.get_lattice_for(solver, point, successor_point)
                }
            };

            successor_lattices.push(after);
        }
        T::visit_region_branch_control_flow_transfer(
            self,
            solver,
            ctx,
            branch,
            predecessor,
            point,
            &successor_lattices,
            before,
        )?;
        Ok(())
    }

    fn visit_call_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        after: &ReadRef<DenseLattice<T::LatticeValue>>,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) -> Result<()> {
        if !solver.config().is_interprocedural {
            return T::visit_external_call(self, solver, ctx, call, after, before);
        }

        let callee = self.resolve_callable(ctx, call);
        let Some(callable) = callee.and_then(|callee| callee.cast::<dyn CallableOpInterface>(ctx))
        else {
            T::set_to_exit_state(self, solver, ctx, before);
            return Ok(());
        };

        // No region means the callee is only *declared* in this module.
        // If that is the case or if the solver is not interprocedural,
        // let the hook handle it.
        let Some(callee_entry_block) = callable
            .callable_region(ctx)
            .and_then(|region| region.deref(ctx).get_entry_block())
        else {
            return T::visit_external_call(self, solver, ctx, call, after, before);
        };

        let callee_entry = ProgramPoint::at_block_start(ctx, callee_entry_block);
        let lattice_at_callee_entry = self.get_lattice_for(
            solver,
            ProgramPoint::before_op(ctx, call.get_operation()),
            callee_entry,
        );
        let lattice_before_call = before;
        T::visit_call_control_flow_transfer(
            self,
            solver,
            ctx,
            call,
            &lattice_at_callee_entry,
            lattice_before_call,
        )
    }

    fn resolve_callable(
        &self,
        ctx: &Context,
        call: &dyn CallOpInterface,
    ) -> Option<Ptr<Operation>> {
        match call.callee(ctx) {
            CallOpCallable::Direct(symbol) => self
                .symbol_table
                .borrow_mut()
                .lookup_symbol_in_nearest_table(ctx, call.get_operation(), &symbol)
                .map(|it| it.get_operation()),
            CallOpCallable::Indirect(value) => value.defining_op(),
        }
    }
}
