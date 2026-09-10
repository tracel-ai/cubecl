use core::{any::TypeId, cell::RefCell, marker::PhantomData};

use alloc::vec::Vec;
use cubecl_ir::{
    dialect::{BlockPtrExt, RegionPtrExt},
    interfaces::control_flow::{
        CallableOpInterface, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
        RegionSuccessor,
    },
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock,
    linked_list::ContainsLinkedList,
    printable::Printable,
    symbol_table::SymbolTableCollection,
    utils::table::{ISet, SmallSet},
};

use crate::analyses::dataflow_solver::{
    AnalysisState, ChangeResult, DataflowSolver, ReadRef, SmallPtrVec, SolverWorkItem, WriteRef,
    dead_code::{CFGEdge, Executable, PredecessorState},
};

use super::{DataflowAnalysis, ProgramPoint};

pub struct SparseForward<T: SparseForwardDataflowAnalysis> {
    _inner: PhantomData<T>,
    symbol_table: RefCell<SymbolTableCollection>,
}

impl<T: SparseForwardDataflowAnalysis> Default for SparseForward<T> {
    fn default() -> Self {
        Self {
            _inner: Default::default(),
            symbol_table: Default::default(),
        }
    }
}

impl<T: SparseForwardDataflowAnalysis> SparseForward<T> {
    pub fn set_all_to_entry_states(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        lattices: &[WriteRef<SparseLattice<T::LatticeValue>>],
    ) {
        for lattice in lattices {
            T::set_to_entry_state(self, solver, ctx, lattice);
        }
    }

    pub fn get_lattice_element<'a>(
        &self,
        solver: &'a DataflowSolver,
        value: Value,
    ) -> ReadRef<'a, SparseLattice<T::LatticeValue>> {
        solver.get_or_create(value)
    }

    pub fn get_lattice_element_mut<'a>(
        &self,
        solver: &'a DataflowSolver,
        value: Value,
    ) -> WriteRef<'a, SparseLattice<T::LatticeValue>> {
        solver.get_or_create_mut(value)
    }

    pub fn get_lattice_element_for<'a>(
        &self,
        solver: &'a DataflowSolver,
        point: ProgramPoint,
        value: Value,
    ) -> ReadRef<'a, SparseLattice<T::LatticeValue>> {
        solver.get_or_create_for::<Self, SparseLattice<T::LatticeValue>>(point, value)
    }

    pub fn join(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        lhs: &WriteRef<SparseLattice<T::LatticeValue>>,
        rhs: &ReadRef<SparseLattice<T::LatticeValue>>,
    ) {
        if lhs != rhs {
            solver.update_state(ctx, lhs, |it| it.join(rhs.deref().value()));
        }
    }

    fn initialize_recursively(
        &self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        self.visit_operation(solver, ctx, op)?;

        for region in op.regions(ctx) {
            for block in region.deref(ctx).iter(ctx) {
                solver
                    .get_or_create::<Executable>(ProgramPoint::at_block_start(ctx, block).into())
                    .deref()
                    .block_content_subscribe::<Self>();
                self.visit_block(solver, ctx, block);
                for op in block.deref(ctx).iter(ctx) {
                    self.initialize_recursively(solver, ctx, op)?;
                }
            }
        }
        Ok(())
    }

    pub fn visit_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        // Exit early on operations with no results.
        if op.deref(ctx).get_num_results() == 0 {
            return Ok(());
        }

        // If the containing block is not executable, bail out.
        if let Some(block) = op.deref(ctx).get_parent_block()
            && !solver
                .get_or_create::<Executable>(ProgramPoint::at_block_start(ctx, block).into())
                .deref()
                .is_live()
        {
            return Ok(());
        }

        let mut result_lattices = SmallPtrVec::with_capacity(op.deref(ctx).get_num_results());
        for result in op.deref(ctx).results() {
            result_lattices.push(self.get_lattice_element_mut(solver, result));
        }

        let dyn_op = op.dyn_op(ctx);
        if let Some(branch) = op_cast::<dyn RegionBranchOpInterface>(&*dyn_op) {
            T::visit_region_successors(
                self,
                solver,
                ctx,
                ProgramPoint::after_op(ctx, op),
                branch,
                RegionSuccessor::AfterOp,
                &result_lattices,
            );
            return Ok(());
        }

        let mut operand_lattices = SmallPtrVec::with_capacity(op.deref(ctx).get_num_operands());
        for operand in op.deref(ctx).operands() {
            let operand_lattice = self.get_lattice_element(solver, operand);
            operand_lattice.deref().use_def_subscribe::<Self>();
            operand_lattices.push(operand_lattice);
        }

        if let Some(call) = op_cast::<dyn CallOpInterface>(&*dyn_op) {
            return T::visit_call_operation(
                self,
                solver,
                ctx,
                call,
                &operand_lattices,
                &result_lattices,
            );
        }

        // Invoke the operation transfer function.
        T::visit_operation(self, solver, ctx, op, &operand_lattices, &result_lattices)
    }

    pub fn visit_block(&self, solver: &DataflowSolver, ctx: &Context, block: Ptr<BasicBlock>) {
        if block.deref(ctx).get_num_arguments() == 0 {
            return;
        }

        let block_start = ProgramPoint::at_block_start(ctx, block);
        if !solver
            .get_or_create::<Executable>(block_start.into())
            .deref()
            .is_live()
        {
            return;
        }

        let mut arg_lattices = SmallPtrVec::with_capacity(block.deref(ctx).get_num_arguments());
        for argument in block.deref(ctx).arguments() {
            arg_lattices.push(self.get_lattice_element_mut(solver, argument));
        }

        // The argument lattices of entry blocks are set by region control-flow or the
        // callgraph.
        if block.is_entry_block(ctx) {
            let parent_region = block.deref(ctx).get_parent_region().unwrap();
            let parent_op = parent_region.deref(ctx).get_parent_op();
            let dyn_op = parent_op.dyn_op(ctx);

            // Check if this block is the entry block of a callable region.
            if let Some(callable) = op_cast::<dyn CallableOpInterface>(&*dyn_op)
                && callable.callable_region(ctx) == Some(parent_region)
            {
                return T::visit_callable_operation(self, solver, ctx, callable, &arg_lattices);
            }

            // Check if the lattices can be determined from region control flow.
            if let Some(branch) = op_cast::<dyn RegionBranchOpInterface>(&*dyn_op) {
                return T::visit_region_successors(
                    self,
                    solver,
                    ctx,
                    block_start,
                    branch,
                    RegionSuccessor::AfterOp,
                    &arg_lattices,
                );
            }

            // All block arguments are non-successor-inputs.
            return T::visit_non_control_flow_arguments(
                self,
                solver,
                ctx,
                parent_op,
                RegionSuccessor::AfterOp,
                &block.arguments(ctx),
                &arg_lattices,
            );
        }

        // Iterate over the predecessors of the non-entry block.
        for r#use in block.uses(ctx) {
            let predecessor = r#use.user_op().deref(ctx).get_parent_block().unwrap();
            // If the edge from the predecessor block to the current block is not live,
            // bail out.
            let edge_executable =
                solver.get_or_create::<Executable>(CFGEdge::new(predecessor, block).into());
            edge_executable.deref().block_content_subscribe::<Self>();
            if !edge_executable.deref().is_live() {
                continue;
            }

            // Check if we can reason about the data-flow from the predecessor.
            if let Some(term) = predecessor.deref(ctx).get_terminator(ctx)
                && let Some(branch) = TraitOp::<dyn BranchOpInterface>::try_from_op(term, ctx)
            {
                let succ_idx = r#use.find_index(ctx);
                let operands = branch.successor_operands(ctx, succ_idx);
                for (idx, lattice) in arg_lattices.iter().enumerate() {
                    let operand = operands.get(idx);
                    if let Some(operand) = operand {
                        let current = self.get_lattice_element_for(solver, block_start, *operand);
                        self.join(solver, ctx, lattice, &current);
                    } else {
                        self.set_all_to_entry_states(solver, ctx, &arg_lattices);
                    }
                }
            } else {
                return self.set_all_to_entry_states(solver, ctx, &arg_lattices);
            }
        }
    }

    pub fn visit_call_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        operand_lattices: &[ReadRef<SparseLattice<T::LatticeValue>>],
        result_lattices: &[WriteRef<SparseLattice<T::LatticeValue>>],
    ) -> Result<()> {
        let call_op = call.get_operation();
        // If the call operation is to an external function, attempt to infer the
        // results from the call arguments.
        let is_external_callable = || {
            let callable = self
                .resolve_callable(ctx, call)
                .and_then(|op| TraitOp::<dyn CallableOpInterface>::try_from_op(op, ctx));
            callable.is_some_and(|callable| callable.callable_region(ctx).is_none())
        };
        if !solver.config().is_interprocedural || is_external_callable() {
            T::visit_external_call(self, solver, ctx, call, operand_lattices, result_lattices);
            return Ok(());
        }

        // Otherwise, the results of a call operation are determined by the
        // callgraph.
        let after_call = ProgramPoint::after_op(ctx, call_op);
        let predecessors =
            solver.get_or_create_for::<Self, PredecessorState>(after_call, after_call);
        if !predecessors.deref().all_predecessors_known() {
            self.set_all_to_entry_states(solver, ctx, result_lattices);
            return Ok(());
        }
        for &predecessor in predecessors.deref().known_predecessors() {
            for (operand, res_lattice) in predecessor.deref(ctx).operands().zip(result_lattices) {
                let current = self.get_lattice_element_for(solver, after_call, operand);
                self.join(solver, ctx, res_lattice, &current)
            }
        }
        Ok(())
    }

    pub fn visit_callable_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
        arg_lattices: &[WriteRef<SparseLattice<T::LatticeValue>>],
    ) {
        let callable_region = callable.callable_region(ctx).unwrap();
        let entry_block = callable_region.deref(ctx).get_entry_block().unwrap();
        let entry_start = ProgramPoint::at_block_start(ctx, entry_block);
        let callsites = solver.get_or_create_for::<Self, PredecessorState>(
            entry_start,
            ProgramPoint::after_op(ctx, callable.get_operation()),
        );
        if !callsites.deref().all_predecessors_known() || !solver.config().is_interprocedural {
            return self.set_all_to_entry_states(solver, ctx, arg_lattices);
        }
        for &callsite in callsites.deref().known_predecessors() {
            let call = TraitOp::<dyn CallOpInterface>::try_from_op(callsite, ctx).unwrap();
            for (operand, lattice) in call.args(ctx).into_iter().zip(arg_lattices) {
                let current = self.get_lattice_element_for(solver, entry_start, operand);
                self.join(solver, ctx, lattice, &current);
            }
        }
    }

    pub fn visit_region_successors(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        successor: RegionSuccessor,
        lattices: &[WriteRef<SparseLattice<T::LatticeValue>>],
    ) {
        let predecessors = solver.get_or_create_for::<Self, PredecessorState>(point, point);
        assert!(
            predecessors.deref().all_predecessors_known(),
            "unexpected unresolved region successors"
        );

        for &op in predecessors.deref().known_predecessors() {
            // Get the incoming successor operands.
            let mut operands = None;

            // Check if the predecessor is the parent op.
            if op == branch.get_operation() {
                operands = Some(branch.entry_successor_operands(ctx, successor));
                // Otherwise, try to deduce the operands from a region return-like op.
            } else if let Some(region_terminator) =
                op_cast::<dyn RegionBranchTerminatorOpInterface>(&*op.dyn_op(ctx))
            {
                operands = Some(region_terminator.successor_operands(ctx, successor));
            }

            // We can't reason about the data-flow.
            let Some(operands) = operands else {
                return self.set_all_to_entry_states(solver, ctx, lattices);
            };

            let predecessors = predecessors.deref();
            let inputs = predecessors.successor_inputs(op);
            assert_eq!(
                inputs.len(),
                operands.len(),
                "expected the same number of successor inputs as operands"
            );

            let mut first_index = 0;
            if inputs.len() != lattices.len() {
                if !point.is_block_start(ctx) {
                    if let Some(first) = inputs.first() {
                        first_index = first.find_index(ctx);
                    }
                    let non_successor_inputs = branch.non_successor_inputs(ctx, successor);
                    let non_successor_input_lattices = non_successor_inputs
                        .iter()
                        .map(|input| self.get_lattice_element_mut(solver, *input))
                        .collect::<Vec<_>>();
                    T::visit_non_control_flow_arguments(
                        self,
                        solver,
                        ctx,
                        branch.get_operation(),
                        successor,
                        &non_successor_inputs,
                        &non_successor_input_lattices,
                    );
                } else {
                    if let Some(first) = inputs.first() {
                        first_index = first.find_index(ctx);
                    }
                    let block = point.block().unwrap();
                    let region = block.deref(ctx).get_parent_region().unwrap();
                    let non_successor_inputs =
                        branch.non_successor_inputs(ctx, RegionSuccessor::Region(region));
                    let non_successor_input_lattices = non_successor_inputs
                        .iter()
                        .map(|input| self.get_lattice_element_mut(solver, *input))
                        .collect::<Vec<_>>();
                    T::visit_non_control_flow_arguments(
                        self,
                        solver,
                        ctx,
                        branch.get_operation(),
                        RegionSuccessor::Region(region),
                        &non_successor_inputs,
                        &non_successor_input_lattices,
                    );
                }
            }

            for (lattice, operand) in lattices.iter().skip(first_index).zip(operands) {
                let other = self.get_lattice_element_for(solver, point, operand);
                self.join(solver, ctx, lattice, &other);
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

impl<T: SparseForwardDataflowAnalysis + 'static> DataflowAnalysis for SparseForward<T> {
    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        for region in root.regions(ctx) {
            for argument in region.arguments(ctx) {
                let lattice = self.get_lattice_element_mut(solver, argument);
                T::set_to_entry_state(self, solver, ctx, &lattice);
            }
        }

        self.initialize_recursively(solver, ctx, root)
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if !point.is_block_start(ctx) {
            return self.visit_operation(solver, ctx, point.prev_op(ctx).unwrap());
        }
        self.visit_block(solver, ctx, point.block().unwrap());
        Ok(())
    }
}

pub trait LatticeValue: Default + PartialEq + Printable + Sized + 'static {
    fn join(&self, rhs: &Self) -> Self;
    fn meet(&self, _rhs: &Self) -> Option<Self> {
        None
    }
}

pub struct SparseLattice<T: LatticeValue> {
    anchor: Value,
    pub value: T,
    dependents: RefCell<ISet<SolverWorkItem>>,
    use_def_subscribers: RefCell<SmallSet<TypeId, 4>>,
}

impl<T: LatticeValue> Printable for SparseLattice<T> {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}: {}", self.anchor.disp(ctx), self.value.disp(ctx))
    }
}

impl<T: LatticeValue> SparseLattice<T> {
    pub fn join(&mut self, rhs: &T) -> ChangeResult {
        let new_value = self.value.join(rhs);
        if new_value == self.value {
            ChangeResult::Unchanged
        } else {
            self.value = new_value;
            ChangeResult::Changed
        }
    }

    pub fn meet(&mut self, rhs: &T) -> ChangeResult {
        let Some(new_value) = self.value.meet(rhs) else {
            return ChangeResult::Unchanged;
        };
        if new_value == self.value {
            ChangeResult::Unchanged
        } else {
            self.value = new_value;
            ChangeResult::Changed
        }
    }

    pub fn use_def_subscribe<A: 'static>(&self) {
        self.use_def_subscribers
            .borrow_mut()
            .insert(TypeId::of::<A>());
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: LatticeValue> AnalysisState for SparseLattice<T> {
    type Anchor = Value;

    fn create(anchor: Value) -> Self {
        Self {
            anchor,
            value: Default::default(),
            dependents: Default::default(),
            use_def_subscribers: Default::default(),
        }
    }

    fn add_dependency<A: 'static>(&self, point: ProgramPoint) {
        self.dependents
            .borrow_mut()
            .insert((point, TypeId::of::<A>()));
    }

    fn on_update(&self, ctx: &Context, solver: &DataflowSolver) {
        for dependent in self.dependents.borrow().iter() {
            solver.enqueue(*dependent);
        }

        for r#use in self.anchor.uses(ctx) {
            let user = r#use.user_op();
            for analysis in self.use_def_subscribers.borrow().iter() {
                solver.enqueue((ProgramPoint::after_op(ctx, user), *analysis));
            }
        }
    }
}

pub trait SparseForwardDataflowAnalysis: Sized + 'static {
    type LatticeValue: LatticeValue;

    /// Verify analysis can be run on the solver. Should be used to verify required analyses are
    /// loaded.
    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        let _ = (solver, ctx, root);
        Ok(())
    }

    #[allow(clippy::result_unit_err)]
    fn visit_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        operands: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        results: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) -> Result<()>;

    fn visit_call_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        operand_lattices: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        result_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) -> Result<()> {
        this.visit_call_operation(solver, ctx, call, operand_lattices, result_lattices)
    }

    fn visit_callable_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
        arg_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        this.visit_callable_operation(solver, ctx, callable, arg_lattices);
    }

    fn visit_region_successors(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        branch: &dyn RegionBranchOpInterface,
        successor: RegionSuccessor,
        lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        this.visit_region_successors(solver, ctx, point, branch, successor, lattices);
    }

    #[allow(clippy::result_unit_err)]
    fn visit_external_call(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
        argument_lattices: &[ReadRef<SparseLattice<Self::LatticeValue>>],
        result_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        let _ = (call, argument_lattices);
        this.set_all_to_entry_states(solver, ctx, result_lattices);
    }

    #[allow(clippy::result_unit_err)]
    fn visit_non_control_flow_arguments(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        successor: RegionSuccessor,
        non_successor_inputs: &[Value],
        non_successor_input_lattices: &[WriteRef<SparseLattice<Self::LatticeValue>>],
    ) {
        let _ = (op, successor, non_successor_inputs);
        this.set_all_to_entry_states(solver, ctx, non_successor_input_lattices);
    }

    fn set_to_entry_state(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<SparseLattice<Self::LatticeValue>>,
    );
}
