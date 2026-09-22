use core::{any::TypeId, cell::RefCell, marker::PhantomData};

use cubecl_ir::{
    dialect::{BlockPtrExt, RegionPtrExt},
    interfaces::control_flow::{
        CallableOpInterface, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
        RegionPredecessor, RegionSuccessor,
    },
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock, graph::walkers::uninterruptible::immutable::walk_op,
    linked_list::ContainsLinkedList, printable::Printable, symbol_table::SymbolTableCollection,
    utils::table::ISet,
};

use crate::analyses::dataflow_solver::{
    AnalysisState, ChangeResult, DataflowAnalysis, ReadRef, SolverWorkItem, WriteRef,
    dead_code::{CFGEdge, DeadCodeAnalysis, Executable, PredecessorState, is_block_live},
};

use super::{DataflowSolver, ProgramPoint};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallControlFlowAction {
    EnterCallee,
    ExitCallee,
    ExternalCallee,
}

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

    pub fn value(&self) -> &T {
        &self.value
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
        solver: &mut DataflowSolver,
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
        predecessor: Ptr<BasicBlock>,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_block_transfer(solver, ctx, block, point, predecessor, before, after);
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_region_branch_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        region_from: Option<usize>,
        region_to: Option<usize>,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_region_branch_control_flow_transfer(
            solver,
            ctx,
            branch,
            region_from,
            region_to,
            before,
            after,
        );
    }

    fn visit_call_control_flow_transfer(
        this: &DenseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        action: CallControlFlowAction,
        before: &ReadRef<DenseLattice<Self::LatticeValue>>,
        after: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_call_control_flow_transfer(solver, ctx, call, action, before, after);
    }
}

pub struct DenseForward<T: DenseForwardDataflowAnalysis> {
    _inner: PhantomData<T>,
    symbol_table: RefCell<SymbolTableCollection>,
}

impl<T: DenseForwardDataflowAnalysis> Default for DenseForward<T> {
    fn default() -> Self {
        Self {
            _inner: Default::default(),
            symbol_table: Default::default(),
        }
    }
}

impl<T: DenseForwardDataflowAnalysis> DataflowAnalysis for DenseForward<T> {
    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<DeadCodeAnalysis>()?;
        T::verify(solver, ctx, root)
    }

    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
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
        solver: &mut DataflowSolver,
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

    #[allow(clippy::too_many_arguments)]
    pub fn visit_block_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _block: Ptr<BasicBlock>,
        _point: ProgramPoint,
        _predecessor: Ptr<BasicBlock>,
        before: &ReadRef<DenseLattice<T::LatticeValue>>,
        after: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.join(solver, ctx, after, before);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn visit_region_branch_control_flow_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _branch: &dyn RegionBranchOpInterface,
        _region_from: Option<usize>,
        _region_to: Option<usize>,
        before: &ReadRef<DenseLattice<T::LatticeValue>>,
        after: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.join(solver, ctx, after, before);
    }

    pub fn visit_call_control_flow_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _call: &dyn CallOpInterface,
        action: CallControlFlowAction,
        before: &ReadRef<DenseLattice<T::LatticeValue>>,
        after: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.join(solver, ctx, after, before);
        // Note that `setToEntryState` may be a "partial fixpoint" for some
        // lattices, e.g., lattices that are lists of maps of other lattices will
        // only set fixpoint for "known" lattices.
        if action == CallControlFlowAction::ExternalCallee {
            T::set_to_entry_state(self, solver, ctx, after);
        }
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

                for callsite in callsites.deref().known_predecessors() {
                    // Get the dense lattice before the callsite.
                    let before = self.get_lattice_for(
                        solver,
                        point,
                        ProgramPoint::before_op(ctx, *callsite),
                    );

                    T::visit_call_control_flow_transfer(
                        self,
                        solver,
                        ctx,
                        op_cast::<dyn CallOpInterface>(&*callsite.dyn_op(ctx)).unwrap(),
                        CallControlFlowAction::EnterCallee,
                        &before,
                        &after,
                    );
                }
                return;
            }

            if let Some(branch) = parent_op.cast::<dyn RegionBranchOpInterface>(ctx) {
                return self.visit_region_branch_operation(solver, ctx, point, &*branch, after);
            }

            return T::set_to_entry_state(self, solver, ctx, &after);
        }

        // Join the state with the state after the block's predecessors.
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
            // Merge in the state from the predecessor's terminator.
            T::visit_block_transfer(
                self,
                solver,
                ctx,
                block,
                point,
                predecessor,
                &before,
                &after,
            );
        }
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
            return T::visit_call_control_flow_transfer(
                self,
                solver,
                ctx,
                call,
                CallControlFlowAction::ExternalCallee,
                before,
                after,
            );
        }

        let predecessors = solver.get_or_create_for::<Self, PredecessorState>(
            ProgramPoint::after_op(ctx, call_op),
            ProgramPoint::after_op(ctx, call_op),
        );
        if !predecessors.deref().all_predecessors_known() {
            return T::set_to_entry_state(self, solver, ctx, after);
        }

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
            let lattice_after_call = after;
            let lattice_at_callee_return = self.get_lattice_for(
                solver,
                ProgramPoint::after_op(ctx, call_op),
                ProgramPoint::after_op(ctx, predecessor),
            );
            T::visit_call_control_flow_transfer(
                self,
                solver,
                ctx,
                call,
                CallControlFlowAction::ExitCallee,
                &lattice_at_callee_return,
                lattice_after_call,
            );
        }
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
            let region_from = match op == branch_op {
                true => None,
                false => {
                    let region = op.deref(ctx).get_parent_region(ctx).unwrap();
                    Some(region.deref(ctx).find_index_in_parent(ctx))
                }
            };

            if point.is_block_start(ctx) {
                let block = point.block().unwrap();
                let region_to = block.deref(ctx).get_parent_region().unwrap();
                let region_to = region_to.deref(ctx).find_index_in_parent(ctx);
                T::visit_region_branch_control_flow_transfer(
                    self,
                    solver,
                    ctx,
                    branch,
                    region_from,
                    Some(region_to),
                    &before,
                    &after,
                );
            } else {
                let parent_op = op.deref(ctx).get_parent_op(ctx).unwrap();
                if parent_op == branch_op || op == branch_op {
                    T::visit_region_branch_control_flow_transfer(
                        self,
                        solver,
                        ctx,
                        branch,
                        region_from,
                        None,
                        &before,
                        &after,
                    );
                } else {
                    self.join(solver, ctx, &after, &before);
                }
            }
        }
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

    /// Verify analysis can be run on the solver. Should be used to verify required analyses are
    /// loaded.
    fn verify(solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        let _ = (solver, ctx, root);
        Ok(())
    }

    fn build_operation_equivalent_lattice_anchor(
        this: &DenseBackward<Self>,
        solver: &mut DataflowSolver,
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

    #[allow(clippy::too_many_arguments)]
    fn visit_block_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        point: ProgramPoint,
        predecessor: Ptr<BasicBlock>,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_block_transfer(solver, ctx, block, point, predecessor, after, before);
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_region_branch_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
        region_from: RegionPredecessor,
        region_to: RegionSuccessor,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_region_branch_control_flow_transfer(
            solver,
            ctx,
            branch,
            region_from,
            region_to,
            after,
            before,
        );
    }

    fn visit_call_control_flow_transfer(
        this: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        action: CallControlFlowAction,
        after: &ReadRef<DenseLattice<Self::LatticeValue>>,
        before: &WriteRef<DenseLattice<Self::LatticeValue>>,
    ) {
        this.visit_call_control_flow_transfer(solver, ctx, call, action, after, before);
    }
}

pub struct DenseBackward<T: DenseBackwardDataflowAnalysis> {
    _inner: PhantomData<T>,
    symbol_table: RefCell<SymbolTableCollection>,
}

impl<T: DenseBackwardDataflowAnalysis> Default for DenseBackward<T> {
    fn default() -> Self {
        Self {
            _inner: Default::default(),
            symbol_table: Default::default(),
        }
    }
}

impl<T: DenseBackwardDataflowAnalysis> DataflowAnalysis for DenseBackward<T> {
    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<DeadCodeAnalysis>()?;
        T::verify(solver, ctx, root)
    }

    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        self.process_operation(solver, ctx, root)?;

        for region in root.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                self.visit_block(solver, ctx, block);
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
        self.visit_block(solver, ctx, point.block().unwrap());
        Ok(())
    }

    fn initialize_equivalent_lattice_anchor(
        &self,
        solver: &mut DataflowSolver,
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

    #[allow(clippy::too_many_arguments)]
    pub fn visit_block_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _block: Ptr<BasicBlock>,
        _point: ProgramPoint,
        _predecessor: Ptr<BasicBlock>,
        after: &ReadRef<DenseLattice<T::LatticeValue>>,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.meet(solver, ctx, before, after);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn visit_region_branch_control_flow_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _branch: &dyn RegionBranchOpInterface,
        _region_from: RegionPredecessor,
        _region_to: RegionSuccessor,
        after: &ReadRef<DenseLattice<T::LatticeValue>>,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.meet(solver, ctx, before, after);
    }

    pub fn visit_call_control_flow_transfer(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        _call: &dyn CallOpInterface,
        action: CallControlFlowAction,
        after: &ReadRef<DenseLattice<T::LatticeValue>>,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        self.meet(solver, ctx, before, after);
        if action == CallControlFlowAction::ExternalCallee {
            T::set_to_exit_state(self, solver, ctx, before);
        }
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
            self.visit_region_branch_operation(
                solver,
                ctx,
                point,
                &*branch,
                RegionPredecessor::Parent,
                &before,
            );
            return Ok(());
        }
        if let Some(call) = op.cast::<dyn CallOpInterface>(ctx) {
            self.visit_call_operation(solver, ctx, &*call, &after, &before);
            return Ok(());
        }

        T::visit_operation(self, solver, ctx, op, &after, &before)
    }

    fn visit_block(&self, solver: &DataflowSolver, ctx: &Context, block: Ptr<BasicBlock>) {
        let parent_region = block.deref(ctx).get_parent_region().unwrap();
        let parent_op = block.deref(ctx).get_parent_op(ctx).unwrap();

        let point = ProgramPoint::at_block_end(ctx, block);
        if !is_block_live::<Self>(solver, ctx, point) {
            return;
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
                    return T::set_to_exit_state(self, solver, ctx, &before);
                }

                for &callsite in callsites.deref().known_predecessors() {
                    let after =
                        self.get_lattice_for(solver, point, ProgramPoint::after_op(ctx, callsite));
                    T::visit_call_control_flow_transfer(
                        self,
                        solver,
                        ctx,
                        &*callsite.cast(ctx).unwrap(),
                        CallControlFlowAction::ExitCallee,
                        &after,
                        &before,
                    );
                }
                return;
            }

            if let Some(branch) = parent_op.cast::<dyn RegionBranchOpInterface>(ctx) {
                let term = block.deref(ctx).get_terminator(ctx).unwrap();
                let terminator = TraitOpPtr::try_from_op(term, ctx).unwrap();
                return self.visit_region_branch_operation(
                    solver,
                    ctx,
                    point,
                    &*branch,
                    RegionPredecessor::Terminator(terminator),
                    &before,
                );
            }

            return T::set_to_exit_state(self, solver, ctx, &before);
        }

        for successor in block.deref(ctx).succs(ctx) {
            let edge = CFGEdge::new(block, successor).into();
            let executable = solver.get_or_create_for::<Self, Executable>(point, edge);
            if !executable.deref().is_live() {
                continue;
            }

            T::visit_block_transfer(
                self,
                solver,
                ctx,
                block,
                point,
                successor,
                &self.get_lattice_for(solver, point, ProgramPoint::at_block_start(ctx, successor)),
                &before,
            );
        }
    }

    fn visit_region_branch_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        predecessor: RegionPredecessor,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        let branch_op = branch.get_operation();
        let successors = branch.successor_regions(ctx, predecessor);
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

            T::visit_region_branch_control_flow_transfer(
                self,
                solver,
                ctx,
                branch,
                predecessor,
                successor,
                &after,
                before,
            );
        }
    }

    fn visit_call_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        after: &ReadRef<DenseLattice<T::LatticeValue>>,
        before: &WriteRef<DenseLattice<T::LatticeValue>>,
    ) {
        if !solver.config().is_interprocedural {
            return T::visit_call_control_flow_transfer(
                self,
                solver,
                ctx,
                call,
                CallControlFlowAction::ExternalCallee,
                after,
                before,
            );
        }

        let callee = self.resolve_callable(ctx, call);
        let Some(callable) = callee.and_then(|callee| callee.cast::<dyn CallableOpInterface>(ctx))
        else {
            return T::set_to_exit_state(self, solver, ctx, before);
        };

        // No region means the callee is only declared in this module.
        // If that is the case or if the solver is not interprocedural,
        // let the hook handle it.
        let Some(callee_entry_block) = callable
            .callable_region(ctx)
            .and_then(|region| region.deref(ctx).get_entry_block())
        else {
            return T::visit_call_control_flow_transfer(
                self,
                solver,
                ctx,
                call,
                CallControlFlowAction::ExternalCallee,
                after,
                before,
            );
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
            CallControlFlowAction::EnterCallee,
            &lattice_at_callee_entry,
            lattice_before_call,
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
