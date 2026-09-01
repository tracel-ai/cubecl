use core::{any::TypeId, cell::RefCell};

use cubecl_ir::{
    interfaces::{
        control_flow::{CallableOpInterface, RegionPredecessor, RegionSuccessor},
        uniformity::{UniformRegionOpInterface, UniformRegionTerminatorOpInterface, Uniformity},
    },
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock,
    graph::HasLabel,
    linked_list::ContainsLinkedList,
    printable::Printable,
    utils::table::{ISet, SmallSet},
};
use smallvec::SmallVec;

use crate::analyses::dataflow_solver::{
    AnalysisState, ChangeResult, DataflowSolver, SmallPtrVec, SolverWorkItem,
    dead_code::{Executable, PredecessorState, is_region_or_callable_return},
    value_uniformity::StrictUniformityLattice,
};

use super::{DataflowAnalysis, ProgramPoint};

pub struct BlockUniformity {
    anchor: Ptr<BasicBlock>,
    value: Uniformity,
    dependents: RefCell<ISet<SolverWorkItem>>,
    subscribers: RefCell<SmallSet<TypeId, 4>>,
}

impl Printable for BlockUniformity {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}: {:?}", self.anchor.label(ctx), self.value)
    }
}

impl BlockUniformity {
    pub fn value(&self) -> Uniformity {
        self.value
    }

    pub fn join(&mut self, other: Uniformity) -> ChangeResult {
        let new_val = self.value.min(other);
        match new_val == self.value {
            true => ChangeResult::Unchanged,
            false => {
                self.value = new_val;
                ChangeResult::Changed
            }
        }
    }

    pub fn block_content_subscribe<A: 'static>(&self) {
        self.subscribers.borrow_mut().insert(TypeId::of::<A>());
    }
}

impl AnalysisState for BlockUniformity {
    type Anchor = Ptr<BasicBlock>;

    fn create(anchor: Self::Anchor) -> Self {
        Self {
            anchor,
            value: Uniformity::Cube,
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

        let block = self.anchor;
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

#[derive(Default)]
pub struct BlockUniformityAnalysis;

impl BlockUniformityAnalysis {
    pub fn set_all_blocks_to_entry_states(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) {
        for region in op.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                let state = solver.get_or_create_mut::<BlockUniformity>(block);
                solver.update_state(ctx, &state, |it| it.join(Uniformity::None));
            }
        }
    }

    fn initialize_recursively(
        &self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Result<()> {
        self.visit_operation(solver, ctx, op)?;

        if op.deref(ctx).num_regions() > 0
            || op.deref(ctx).get_num_successors() > 0
            || is_region_or_callable_return(ctx, op)
            || op.impls::<dyn CallOpInterface>(ctx)
        {
            if let Some(block) = op.deref(ctx).get_parent_block() {
                solver
                    .get_or_create::<BlockUniformity>(block)
                    .deref()
                    .block_content_subscribe::<Self>();
            }
            self.visit(solver, ctx, ProgramPoint::after_op(ctx, op))?;
        }

        for region in op.regions(ctx) {
            for block in region.deref(ctx).iter(ctx) {
                let executable = solver
                    .get_or_create::<Executable>(ProgramPoint::at_block_start(ctx, block).into());
                executable.deref().block_content_subscribe::<Self>();
                let uniformity = solver.get_or_create::<BlockUniformity>(block);
                uniformity.deref().block_content_subscribe::<Self>();

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
        let dyn_op = op.dyn_op(ctx);

        if let Some(branch) = op_cast::<dyn UniformRegionOpInterface>(&*dyn_op) {
            self.visit_uniform_region_op(solver, ctx, branch);
        } else if let Some(term) = op_cast::<dyn UniformRegionTerminatorOpInterface>(&*dyn_op)
            && let Some(parent_op) = op.deref(ctx).get_parent_op(ctx)
            && parent_op.impls::<dyn UniformRegionOpInterface>(ctx)
        {
            self.visit_uniform_region_terminator_op(solver, ctx, term);
        } else if let Some(callable) = op_cast::<dyn CallableOpInterface>(&*dyn_op) {
            self.visit_callable_operation_uniformity(solver, ctx, callable);
        } else if op.deref(ctx).num_regions() > 0 {
            self.set_all_blocks_to_entry_states(solver, ctx, op);
        }

        Ok(())
    }

    fn visit_uniform_region_op(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        branch: &dyn UniformRegionOpInterface,
    ) {
        let op = branch.get_operation();
        let block = op
            .deref(ctx)
            .get_parent_block()
            .expect("Should have parent");
        let parent_uniformity = solver
            .get_or_create::<BlockUniformity>(block)
            .deref()
            .value();

        let mut operand_uniformity = SmallPtrVec::with_capacity(op.deref(ctx).get_num_operands());
        for operand in op.deref(ctx).operands() {
            let lattice = solver.get_or_create::<StrictUniformityLattice>(operand);
            lattice.deref().use_def_subscribe::<Self>();
            operand_uniformity.push(lattice.deref().value().0);
        }
        let successors = branch.successor_regions(ctx, RegionPredecessor::Parent);
        let uniformity = branch.entry_successor_region_uniformity(ctx, &operand_uniformity);
        let successors =
            Self::filter_successors(solver, ctx, op, successors.into_iter().zip(uniformity));

        // Const-folded successors inherit uniformity
        if successors.len() == 1 {
            let (entry, _) = successors[0];
            let state = solver.get_or_create_mut::<BlockUniformity>(entry);
            solver.update_state(ctx, &state, |it| it.join(parent_uniformity));
            return;
        }

        for (entry, uniformity) in successors {
            let uniformity = uniformity.min(parent_uniformity);
            let state = solver.get_or_create_mut::<BlockUniformity>(entry);
            solver.update_state(ctx, &state, |it| it.join(uniformity));
        }
    }

    fn filter_successors(
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        successors: impl Iterator<Item = (RegionSuccessor, Uniformity)>,
    ) -> SmallVec<[(Ptr<BasicBlock>, Uniformity); 4]> {
        successors
            .filter_map(|(successor, uniformity)| match successor {
                RegionSuccessor::Region(region) => {
                    let entry = region.deref(ctx).get_entry_block()?;
                    let executable = solver.get_or_create_for::<Self, Executable>(
                        ProgramPoint::after_op(ctx, op),
                        ProgramPoint::at_block_start(ctx, entry).into(),
                    );
                    match executable.deref().is_live() {
                        true => Some((entry, uniformity)),
                        false => None,
                    }
                }
                RegionSuccessor::AfterOp => None,
            })
            .collect()
    }

    fn visit_uniform_region_terminator_op(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        term: &dyn UniformRegionTerminatorOpInterface,
    ) {
        let op = term.get_operation();
        let block = op
            .deref(ctx)
            .get_parent_block()
            .expect("Should have parent");
        let parent_uniformity = solver
            .get_or_create::<BlockUniformity>(block)
            .deref()
            .value();

        let mut operand_uniformity = SmallPtrVec::with_capacity(op.deref(ctx).get_num_operands());
        for operand in op.deref(ctx).operands() {
            let lattice = solver.get_or_create::<StrictUniformityLattice>(operand);
            lattice.deref().use_def_subscribe::<Self>();
            operand_uniformity.push(lattice.deref().value().0);
        }
        let successors = term.all_successor_regions(ctx);
        let uniformity = term.successor_region_uniformity(ctx, &operand_uniformity);
        let successors =
            Self::filter_successors(solver, ctx, op, successors.into_iter().zip(uniformity));

        // Const-folded successors inherit uniformity
        if successors.len() == 1 {
            let (entry, _) = successors[0];
            let state = solver.get_or_create_mut::<BlockUniformity>(entry);
            solver.update_state(ctx, &state, |it| it.join(parent_uniformity));
            return;
        }

        for (entry, uniformity) in successors {
            let uniformity = uniformity.min(parent_uniformity);
            let state = solver.get_or_create_mut::<BlockUniformity>(entry);
            solver.update_state(ctx, &state, |it| it.join(uniformity));
        }
    }

    fn visit_callable_operation_uniformity(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        callable: &dyn CallableOpInterface,
    ) {
        let callable_region = callable.callable_region(ctx).unwrap();
        let entry_block = callable_region.deref(ctx).get_entry_block().unwrap();
        let entry_start = ProgramPoint::at_block_start(ctx, entry_block);
        let callsites = solver.get_or_create_for::<Self, PredecessorState>(
            entry_start,
            ProgramPoint::after_op(ctx, callable.get_operation()),
        );
        if !callsites.deref().all_predecessors_known()
            || callable_region.deref(ctx).iter(ctx).count() > 1
        {
            return self.set_all_blocks_to_entry_states(solver, ctx, callable.get_operation());
        }

        let lattice = solver.get_or_create_mut::<BlockUniformity>(entry_block);
        for &callsite in callsites.deref().known_predecessors() {
            let block = callsite.deref(ctx).get_parent_block().unwrap();
            let callsite_lattice =
                solver.get_or_create_for::<Self, BlockUniformity>(entry_start, block);
            let callsite_uniformity = callsite_lattice.deref().value();
            solver.update_state(ctx, &lattice, |it| it.join(callsite_uniformity));
        }
    }
}

impl DataflowAnalysis for BlockUniformityAnalysis {
    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        for region in root.regions(ctx) {
            if region.deref(ctx).iter(ctx).count() > 1 {
                self.set_all_blocks_to_entry_states(solver, ctx, root);
            }
        }

        self.initialize_recursively(solver, ctx, root)
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if !point.is_block_start(ctx) {
            return self.visit_operation(solver, ctx, point.prev_op(ctx).unwrap());
        }
        Ok(())
    }
}
