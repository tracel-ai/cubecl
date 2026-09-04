use core::{any::TypeId, cell::RefCell, hash::Hash};

use alloc::string::ToString;
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::BlockPtrExt,
    interfaces::{
        ReturnLike,
        control_flow::{
            CallableOpInterface, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
            RegionSuccessor, SymbolOpInterface,
        },
    },
    prelude::*,
};
use derive_more::From;
use derive_new::new;
use itertools::Itertools;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    dyn_clone,
    graph::{ControlFlowGraph, HasLabel},
    linked_list::ContainsLinkedList,
    opts::constants::BranchOpFoldInterface,
    printable::Printable,
    symbol_table::SymbolTableCollection,
    utils::table::{ISet, SmallSet},
};
use smallvec::SmallVec;

use crate::analyses::{
    dataflow_solver::{
        AnalysisState, ChangeResult, DataflowAnalysis, ProgramPoint, SolverWorkItem,
        sccp::{ConstantLattice, ConstantValue, SparseConstantPropagationAnalysis},
    },
    symbol_table::walk_symbol_tables,
};

use super::DataflowSolver;

#[derive(PartialEq, Eq, Hash, Clone, Copy, new)]
pub struct CFGEdge {
    pub from: Ptr<BasicBlock>,
    pub to: Ptr<BasicBlock>,
}

impl Printable for CFGEdge {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "CFGEdge({} -> {})",
            self.from.label(ctx),
            self.to.label(ctx)
        )
    }
}

#[derive(PartialEq, Eq, Hash, From, Clone)]
pub enum ControlFlowAnchor {
    ProgramPoint(ProgramPoint),
    Edge(CFGEdge),
}

impl Printable for ControlFlowAnchor {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            ControlFlowAnchor::ProgramPoint(program_point) => {
                Printable::fmt(program_point, ctx, state, f)
            }
            ControlFlowAnchor::Edge(edge) => Printable::fmt(edge, ctx, state, f),
        }
    }
}

pub struct Executable {
    anchor: ControlFlowAnchor,
    live: bool,
    dependents: RefCell<ISet<SolverWorkItem>>,
    subscribers: RefCell<SmallSet<TypeId, 4>>,
}

impl Printable for Executable {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{}: {}",
            self.anchor.disp(ctx),
            match self.live {
                true => "Executable::Live",
                false => "Executable::Dead",
            }
        )
    }
}

impl Executable {
    pub fn is_live(&self) -> bool {
        self.live
    }

    pub fn set_to_live(&mut self) -> ChangeResult {
        match self.live {
            true => ChangeResult::Unchanged,
            false => {
                self.live = true;
                ChangeResult::Changed
            }
        }
    }

    pub fn block_content_subscribe<A: 'static>(&self) {
        self.subscribers.borrow_mut().insert(TypeId::of::<A>());
    }
}

impl AnalysisState for Executable {
    type Anchor = ControlFlowAnchor;

    fn create(anchor: Self::Anchor) -> Self {
        Self {
            anchor,
            live: false,
            dependents: Default::default(),
            subscribers: Default::default(),
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

        match &self.anchor {
            ControlFlowAnchor::ProgramPoint(point) => {
                if point.is_block_start(ctx) {
                    let block = point.block().unwrap();
                    for analysis in self.subscribers.borrow().iter() {
                        solver.enqueue((ProgramPoint::at_block_start(ctx, block), *analysis));
                    }
                    for analysis in self.subscribers.borrow().iter() {
                        for op in block.deref(ctx).iter(ctx) {
                            solver.enqueue((ProgramPoint::after_op(ctx, op), *analysis));
                        }
                    }
                }
            }
            ControlFlowAnchor::Edge(edge) => {
                for analysis in self.subscribers.borrow().iter() {
                    solver.enqueue((ProgramPoint::at_block_start(ctx, edge.to), *analysis));
                }
            }
        }
    }
}

pub struct PredecessorState {
    anchor: ProgramPoint,
    all_known: bool,
    known_predecessors: SmallSet<Ptr<Operation>, 4>,
    successor_inputs: HashMap<Ptr<Operation>, Vec<Value>>,
    dependents: RefCell<ISet<SolverWorkItem>>,
}

impl Printable for PredecessorState {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{}: PredecessorState(all_known: {}, known_predecessors: [{}], successor_inputs: {{{}}})",
            self.anchor.disp(ctx),
            self.all_known,
            self.known_predecessors
                .iter()
                .map(|it| it.disp(ctx).to_string())
                .join(", "),
            self.successor_inputs
                .iter()
                .map(|(op, values)| {
                    let values = values.iter().map(|it| it.disp(ctx).to_string()).join(", ");
                    alloc::format!("{}: [{}]", op.disp(ctx), values)
                })
                .join(", ")
        )
    }
}

impl PredecessorState {
    pub fn all_predecessors_known(&self) -> bool {
        self.all_known
    }

    pub fn set_has_unknown_predecessors(&mut self) -> ChangeResult {
        match self.all_known {
            true => {
                self.all_known = false;
                ChangeResult::Changed
            }
            false => ChangeResult::Unchanged,
        }
    }

    pub fn join(&mut self, call: Ptr<Operation>) -> ChangeResult {
        match self.known_predecessors.insert(call) {
            true => ChangeResult::Changed,
            false => ChangeResult::Unchanged,
        }
    }

    pub fn join_with_inputs(&mut self, call: Ptr<Operation>, inputs: Vec<Value>) -> ChangeResult {
        let changed = self.join(call);
        match self.successor_inputs.insert(call, inputs) {
            Some(_) => changed,
            None => ChangeResult::Changed,
        }
    }

    pub fn known_predecessors(&self) -> &SmallSet<Ptr<Operation>, 4> {
        &self.known_predecessors
    }

    pub fn successor_inputs(&self, predecessor: Ptr<Operation>) -> &[Value] {
        if let Some(inputs) = self.successor_inputs.get(&predecessor) {
            inputs
        } else {
            &[]
        }
    }
}

impl AnalysisState for PredecessorState {
    type Anchor = ProgramPoint;

    fn create(anchor: Self::Anchor) -> Self {
        Self {
            anchor,
            all_known: true,
            known_predecessors: Default::default(),
            successor_inputs: Default::default(),
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

#[derive(Default)]
pub struct DeadCodeAnalysis {
    symbol_table: RefCell<SymbolTableCollection>,
    analysis_scope: Option<Ptr<Operation>>,
}

impl DeadCodeAnalysis {
    pub fn initialize_symbol_callables(
        &mut self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) {
        self.analysis_scope = Some(root);
        let all_uses_visible = root.deref(ctx).get_parent_block().is_none();
        walk_symbol_tables(
            ctx,
            &mut (self, solver),
            root,
            all_uses_visible,
            |ctx, (this, solver), symbol_table, all_uses_visible| {
                this.symbol_table
                    .borrow_mut()
                    .get_symbol_table(ctx, dyn_clone::clone_box(symbol_table));
                let symbol_table_block = symbol_table.get_body(ctx, 0);

                let mut found_symbol_callable = false;
                for callable in
                    symbol_table_block.ops_with_interface::<dyn CallableOpInterface>(ctx)
                {
                    let Some(_callable_region) = callable.callable_region(ctx) else {
                        continue;
                    };
                    let Some(symbol) = op_cast::<dyn SymbolOpInterface>(callable.dyn_op()) else {
                        continue;
                    };

                    if symbol.is_public(ctx) || (!all_uses_visible && symbol.is_nested(ctx)) {
                        let state = solver.get_or_create_mut::<PredecessorState>(
                            ProgramPoint::after_op(ctx, callable.get_operation()),
                        );
                        solver.update_state(ctx, &state, |it| it.set_has_unknown_predecessors());
                    }
                    found_symbol_callable = true;
                }

                if !found_symbol_callable {
                    return;
                }

                let uses = symbol_table_block
                    .ops_with_interface::<dyn SymbolUserOpInterface>(ctx)
                    .flat_map(|user| {
                        let symbols = user.used_symbols(ctx);
                        symbols.into_iter().map(move |sym| (user.clone(), sym))
                    });
                for (user, used) in uses {
                    if op_impls::<dyn CallOpInterface>(user.dyn_op()) {
                        continue;
                    }

                    let Some(symbol) = symbol_table.lookup(ctx, &used) else {
                        continue;
                    };
                    let state = solver
                        .get_or_create_mut::<PredecessorState>(ProgramPoint::after_op(ctx, symbol));
                    solver.update_state(ctx, &state, |it| it.set_has_unknown_predecessors());
                }
            },
        );
    }

    pub fn initialize_recursively(
        &self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        if op.deref(ctx).num_regions() > 0
            || op.deref(ctx).get_num_successors() > 0
            || is_region_or_callable_return(ctx, op)
            || op.impls::<dyn CallOpInterface>(ctx)
        {
            if let Some(block) = op.deref(ctx).get_parent_block() {
                let point = ProgramPoint::at_block_start(ctx, block);
                solver
                    .get_or_create::<Executable>(point.into())
                    .deref()
                    .block_content_subscribe::<Self>();
            }
            self.visit(solver, ctx, ProgramPoint::after_op(ctx, op))?;
        }

        if op.deref(ctx).num_regions() > 0 {
            for region in op.regions(ctx) {
                for block in region.deref(ctx).iter(ctx) {
                    for nested_op in block.deref(ctx).iter(ctx) {
                        self.initialize_recursively(solver, ctx, nested_op)?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn mark_edge_live(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        from: Ptr<BasicBlock>,
        to: Ptr<BasicBlock>,
    ) {
        let state =
            solver.get_or_create_mut::<Executable>(ProgramPoint::at_block_start(ctx, to).into());
        solver.update_state(ctx, &state, |it| it.set_to_live());
        let edge_state = solver.get_or_create_mut::<Executable>(CFGEdge::new(from, to).into());
        solver.update_state(ctx, &edge_state, |it| it.set_to_live());
    }

    pub fn mark_entry_blocks_live(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) {
        for region in op.regions(ctx) {
            let Some(entry) = region.deref(ctx).get_entry_block() else {
                continue;
            };
            let state = solver
                .get_or_create_mut::<Executable>(ProgramPoint::at_block_start(ctx, entry).into());
            solver.update_state(ctx, &state, |it| it.set_to_live());
        }
    }

    pub fn visit_call_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        call: &dyn CallOpInterface,
    ) {
        let call_op = call.get_operation();
        let callable = match call.callee(ctx) {
            CallOpCallable::Direct(identifier) => self
                .symbol_table
                .borrow_mut()
                .lookup_symbol_in_nearest_table(ctx, call_op, &identifier)
                .map(|op| op.get_operation()),
            CallOpCallable::Indirect(value) => value.defining_op(),
        };

        // A call to a externally-defined callable has unknown predecessors.
        let is_external_callable = |op: Ptr<Operation>| {
            if !self.analysis_scope.unwrap().is_ancestor_of(ctx, op) {
                return true;
            }
            if let Some(callable) = TraitOp::<dyn CallableOpInterface>::try_from_op(op, ctx) {
                return callable.callable_region(ctx).is_none();
            }
            false
        };

        if let Some(callable) = callable
            && callable.impls::<dyn SymbolOpInterface>(ctx)
            && !is_external_callable(callable)
        {
            let callsites =
                solver.get_or_create_mut::<PredecessorState>(ProgramPoint::after_op(ctx, callable));
            solver.update_state(ctx, &callsites, |it| it.join(call_op));
        } else {
            let predecessors =
                solver.get_or_create_mut::<PredecessorState>(ProgramPoint::after_op(ctx, call_op));
            solver.update_state(ctx, &predecessors, |it| it.set_has_unknown_predecessors());
        }
    }

    fn operand_values(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<SmallVec<[Option<AttrObj>; 8]>> {
        let mut operands = SmallVec::with_capacity(op.deref(ctx).get_num_operands());
        for operand in op.deref(ctx).operands() {
            let cv = solver.get_or_create::<ConstantLattice>(operand);
            cv.deref().use_def_subscribe::<Self>();
            // Not yet initialized, skip until SCCP runs
            if matches!(cv.deref().value(), ConstantValue::Uninitialized) {
                return None;
            }
            operands.push(cv.deref().value().constant_attr().cloned());
        }
        Some(operands)
    }

    pub fn visit_branch_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn BranchOpFoldInterface,
    ) {
        let op = branch.get_operation();
        let block = op.deref(ctx).get_parent_block().unwrap();
        let Some(operands) = self.operand_values(solver, ctx, op) else {
            return;
        };

        for successor in branch.check_fold(ctx, &operands) {
            self.mark_edge_live(solver, ctx, block, successor);
        }
    }

    pub fn visit_region_branch_operation(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn RegionBranchOpInterface,
    ) {
        let op = branch.get_operation();
        let Some(operands) = self.operand_values(solver, ctx, op) else {
            return;
        };
        let successors = branch.entry_successor_regions(ctx, &operands);
        self.visit_region_branch_edges(solver, ctx, branch, op, successors);
    }

    pub fn visit_region_terminator(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        branch: &dyn RegionBranchOpInterface,
    ) {
        let Some(operands) = self.operand_values(solver, ctx, op) else {
            return;
        };

        let dyn_op = op.dyn_op(ctx);
        if let Some(terminator) = op_cast::<dyn RegionBranchTerminatorOpInterface>(&*dyn_op) {
            let successors = terminator.successor_regions(ctx, &operands);
            self.visit_region_branch_edges(solver, ctx, branch, op, successors);
        }
    }

    fn visit_region_branch_edges(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        region_branch_op: &dyn RegionBranchOpInterface,
        predecessor_op: Ptr<Operation>,
        successors: Vec<RegionSuccessor>,
    ) {
        for successor in successors {
            // The successor can be either an entry block or an operation to resume
            // after.
            // Skip empty regions — they have no entry block to mark executable.
            let point = match successor {
                RegionSuccessor::Region(region) if let Some(entry) = region.entry_node(ctx) => {
                    ProgramPoint::at_block_start(ctx, entry)
                }
                RegionSuccessor::Region(_) => {
                    continue;
                }
                RegionSuccessor::AfterOp => {
                    ProgramPoint::after_op(ctx, region_branch_op.get_operation())
                }
            };

            // Mark the entry block as executable.
            let state = solver.get_or_create_mut::<Executable>(point.into());
            solver.update_state(ctx, &state, |it| it.set_to_live());

            // Add the region branch predecessor.
            let predecessors = solver.get_or_create_mut::<PredecessorState>(point);
            solver.update_state(ctx, &predecessors, |it| {
                it.join_with_inputs(
                    predecessor_op,
                    region_branch_op.successor_inputs(ctx, successor),
                )
            });
        }
    }

    fn visit_callable_terminator(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        callable: &dyn CallableOpInterface,
    ) {
        let callsites = solver.get_or_create_for::<Self, PredecessorState>(
            ProgramPoint::after_op(ctx, op),
            ProgramPoint::after_op(ctx, callable.get_operation()),
        );
        let can_resolve = op.impls::<dyn ReturnLike>(ctx);
        for predecessor in callsites.deref().known_predecessors.iter().copied() {
            let predecessors = solver
                .get_or_create_mut::<PredecessorState>(ProgramPoint::after_op(ctx, predecessor));
            if can_resolve {
                solver.update_state(ctx, &predecessors, |it| it.join(op));
            } else {
                // If the terminator is not a return-like, then conservatively assume we
                // can't resolve the predecessor.
                solver.update_state(ctx, &predecessors, |it| it.set_has_unknown_predecessors());
            }
        }
    }
}

pub(crate) fn is_region_or_callable_return(ctx: &Context, op: Ptr<Operation>) -> bool {
    let Some(block) = op.deref(ctx).get_parent_block() else {
        return false;
    };
    let parent_op = block.deref(ctx).get_parent_op(ctx).unwrap();
    let parent_region_or_callable = parent_op.impls::<dyn RegionBranchOpInterface>(ctx)
        || parent_op.impls::<dyn CallOpInterface>(ctx);
    op.deref(ctx).get_num_successors() == 0
        && parent_region_or_callable
        && block.deref(ctx).get_terminator(ctx) == Some(op)
}

impl DataflowAnalysis for DeadCodeAnalysis {
    fn verify(&self, solver: &DataflowSolver, _ctx: &Context, _root: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<SparseConstantPropagationAnalysis>()
    }

    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        for region in root.regions(ctx) {
            let Some(entry) = region.entry_node(ctx) else {
                continue;
            };
            let state = solver
                .get_or_create_mut::<Executable>(ProgramPoint::at_block_start(ctx, entry).into());
            solver.update_state(ctx, &state, |it| it.set_to_live());
        }

        if root.impls::<dyn CallableOpInterface>(ctx) {
            let state =
                solver.get_or_create_mut::<PredecessorState>(ProgramPoint::after_op(ctx, root));
            solver.update_state(ctx, &state, |it| it.set_has_unknown_predecessors());
        }

        self.initialize_symbol_callables(solver, ctx, root);
        self.initialize_recursively(solver, ctx, root)
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        let Some(op) = point.prev_op(ctx) else {
            return Ok(());
        };

        // If the parent block is not executable, there is nothing to do.
        if let Some(block) = op.deref(ctx).get_parent_block()
            && !solver
                .get_or_create::<Executable>(ProgramPoint::at_block_start(ctx, block).into())
                .deref()
                .is_live()
        {
            return Ok(());
        }

        let dyn_op = op.dyn_op(ctx);

        // We have a live call op. Add this as a live predecessor of the callee.
        if let Some(call) = op_cast::<dyn CallOpInterface>(&*dyn_op) {
            self.visit_call_operation(solver, ctx, call);
        }

        if op.deref(ctx).num_regions() > 0 {
            if let Some(branch) = op_cast::<dyn RegionBranchOpInterface>(&*dyn_op) {
                self.visit_region_branch_operation(solver, ctx, branch);
            } else if op_impls::<dyn CallableOpInterface>(&*dyn_op) {
                let callsites = solver.get_or_create_for::<Self, PredecessorState>(
                    ProgramPoint::after_op(ctx, op),
                    ProgramPoint::after_op(ctx, op),
                );

                // If the callsites could not be resolved or are known to be non-empty,
                // mark the callable as executable.
                if !callsites.deref().all_predecessors_known()
                    || !callsites.deref().known_predecessors.is_empty()
                {
                    self.mark_entry_blocks_live(solver, ctx, op);
                }

                // Otherwise, conservatively mark all entry blocks as executable.
            } else {
                self.mark_entry_blocks_live(solver, ctx, op);
            }
        }

        if is_region_or_callable_return(ctx, op) {
            let parent = op.deref(ctx).get_parent_op(ctx).unwrap().dyn_op(ctx);
            // Check if we can reason about the control-flow.
            if let Some(branch) = op_cast::<dyn RegionBranchOpInterface>(&*parent) {
                self.visit_region_terminator(solver, ctx, op, branch);
            } else if let Some(callable) = op_cast::<dyn CallableOpInterface>(&*parent) {
                self.visit_callable_terminator(solver, ctx, op, callable);
            }
        }

        // Visit the successors.
        if op.deref(ctx).get_num_successors() > 0 {
            // Check if we can reason about the control-flow.
            if let Some(branch) = op_cast::<dyn BranchOpFoldInterface>(&*dyn_op) {
                self.visit_branch_operation(solver, ctx, branch);
                // Otherwise, conservatively mark all successors as executable.
            } else {
                for successor in op.deref(ctx).successors() {
                    let block = op.deref(ctx).get_parent_block().unwrap();
                    self.mark_edge_live(solver, ctx, block, successor);
                }
            }
        }

        Ok(())
    }
}
