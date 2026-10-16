use core::fmt;

use alloc::collections::VecDeque;
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::BlockPtrExt,
    interfaces::{
        MaterializableOp,
        control_flow::{
            MutableRegionBranchOpInterface, MutableRegionBranchTerminatorOpInterface,
            RegionBranchOpInterface, RegionPredecessor, RegionSuccessor,
        },
    },
    prelude::*,
};
use hi_sparse_bitset::ops;
use pliron::{
    basic_block::BasicBlock,
    graph::{
        ControlFlowGraph, HasLabel, dominance::DomInfo,
        walkers::uninterruptible::immutable::walk_op,
    },
    linked_list::ContainsLinkedList,
    operation::OpDbg,
    opts::dce::dce,
    printable::{self, Printable, indented_nl},
    region::Region,
    utils::table::IMap,
};

use crate::{
    BitSetExt, SparseBitSet,
    analyses::dataflow_solver::{
        DataflowSolver, ProgramPoint,
        pre::{
            AnticipatedMode, Available, InsertMode, compute_lcm_analyses,
            compute_pruned_insertion_points,
        },
        value_numbering::ValueNumberLattice,
    },
};

pub fn run_pre(
    ctx: &mut Context,
    op: Ptr<Operation>,
    rewriter: &mut impl Rewriter,
    antic_mode: AnticipatedMode,
    insert_mode: InsertMode,
) -> Result<IRStatus> {
    let mut solver = compute_lcm_analyses(ctx, op, antic_mode)?;
    let insertions = compute_pruned_insertion_points(ctx, op, &mut solver, insert_mode)?;

    let mut status = IRStatus::Unchanged;
    let antileaders = collect_antileaders(ctx, op, &solver, &insertions);
    let leaders = &mut compute_leaders(ctx, op, &solver, &insertions)?;

    status |= insert_expressions(ctx, &solver, insertions, &antileaders, leaders, rewriter);
    status |= replace_redundant_values(ctx, &solver, op, leaders, rewriter);

    leaders.link_materialized(ctx);

    status |= dce(op, ctx)?;

    Ok(status)
}

type RegionMerge = (Ptr<Operation>, RegionSuccessor);

#[derive(Default)]
struct Leaders {
    leaders: HashMap<ProgramPoint, HashMap<u64, Leader>>,
    materialized: HashMap<Leader, Value>,
    blocks_to_link: IMap<Ptr<BasicBlock>, Vec<(u64, Value)>>,
    regions_to_link: IMap<RegionMerge, Vec<(u64, Value)>>,
}

impl Printable for Leaders {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str("leaders: {")?;
        {
            let _indent = state.indent();
            for (point, leaders) in self.leaders.iter() {
                write!(f, "{}{}: {{", indented_nl(state), point.disp(ctx))?;
                {
                    let _indent = state.indent();
                    for (expr, leader) in leaders {
                        write!(f, "{}e{expr}: {},", indented_nl(state), leader.disp(ctx))?;
                    }
                }
                write!(f, "{}}},", indented_nl(state))?;
            }
        }
        f.write_str("\n}\n")?;
        f.write_str("materialized: {")?;
        {
            let _indent = state.indent();
            for (leader, value) in self.materialized.iter() {
                write!(
                    f,
                    "{}{}: {},",
                    indented_nl(state),
                    leader.disp(ctx),
                    value.disp(ctx)
                )?;
            }
        }
        f.write_str("\n}")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Leader {
    Material(Value),
    RegionArg(Ptr<Operation>, RegionSuccessor),
    BlockArg(Ptr<BasicBlock>),
    Insertion(ProgramPoint, u64),
}

impl Printable for Leader {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Leader::Material(value) => value.fmt(ctx, state, f),
            Leader::RegionArg(op, region) => write!(
                f,
                "RegionArg({}, {})",
                OpDbg { op: *op, ctx },
                region.disp(ctx)
            ),
            Leader::BlockArg(block) => write!(f, "BlockArg({})", block.label(ctx)),
            Leader::Insertion(point, expr) => write!(f, "Insertion({}, e{expr})", point.disp(ctx)),
        }
    }
}

impl Leaders {
    fn materialize(
        &mut self,
        ctx: &mut Context,
        at: &ProgramPoint,
        expression: u64,
        ty: TypeHandle,
    ) -> Value {
        self.try_materialize(ctx, at, expression, ty).unwrap()
    }

    fn try_materialize(
        &mut self,
        ctx: &mut Context,
        at: &ProgramPoint,
        expr: u64,
        ty: TypeHandle,
    ) -> Option<Value> {
        let leader = self.leaders[at][&expr];
        match leader {
            Leader::Material(value) => Some(value),
            _ if let Some(materialized) = self.materialized.get(&leader) => Some(*materialized),
            Leader::RegionArg(region_op, successor) => {
                // Should already be validated by analysis
                let branch_op = region_op
                    .cast::<dyn MutableRegionBranchOpInterface>(ctx)
                    .unwrap();
                let arg = branch_op.add_successor_input(ctx, successor, ty);
                self.materialized.insert(leader, arg);
                self.regions_to_link
                    .entry((region_op, successor))
                    .or_default()
                    .push((expr, arg));
                Some(arg)
            }
            Leader::BlockArg(block) => {
                let idx = BasicBlock::push_argument(block, ctx, ty);
                let arg = block.deref(ctx).get_argument(idx);
                self.materialized.insert(leader, arg);
                self.blocks_to_link
                    .entry(block)
                    .or_default()
                    .push((expr, arg));
                Some(arg)
            }
            Leader::Insertion(..) => None,
        }
    }

    fn link_materialized(&mut self, ctx: &mut Context) {
        while let Some((block, to_link)) = self.blocks_to_link.pop() {
            for (expression, arg) in to_link {
                let ty = arg.get_type(ctx);
                for pred in block.uses(ctx) {
                    let block = pred.user_op().deref(ctx).get_parent_block().unwrap();
                    let at = ProgramPoint::at_block_end(ctx, block);
                    let leader = self.materialize(ctx, &at, expression, ty);
                    // Should be validated by analysis
                    let branch = pred.user_op().cast::<dyn BranchOpInterface>(ctx).unwrap();
                    branch.add_successor_operand(ctx, pred.find_index(ctx), leader);
                }
            }
        }
        while let Some(((op, successor), to_link)) = self.regions_to_link.pop() {
            let region_op = op.cast::<dyn MutableRegionBranchOpInterface>(ctx).unwrap();
            for (expression, arg) in to_link {
                let ty = arg.get_type(ctx);
                for predecessor in region_op.predecessors(ctx, successor) {
                    match predecessor {
                        RegionPredecessor::Parent => {
                            let at = ProgramPoint::before_op(ctx, op);
                            let leader = self.materialize(ctx, &at, expression, ty);
                            region_op.add_entry_successor_operand(ctx, successor, leader);
                        }
                        RegionPredecessor::Terminator(term) => {
                            let term = term.get_operation();
                            let block = term.deref(ctx).get_parent_block().unwrap();
                            let term = term
                                .cast::<dyn MutableRegionBranchTerminatorOpInterface>(ctx)
                                .unwrap();
                            let at = ProgramPoint::at_block_end(ctx, block);
                            let leader = self.materialize(ctx, &at, expression, ty);
                            term.add_successor_operand(ctx, successor, leader);
                        }
                    }
                }
            }
        }
    }
}

fn compute_leaders(
    ctx: &Context,
    op: Ptr<Operation>,
    solver: &DataflowSolver,
    insertions: &IMap<ProgramPoint, SparseBitSet>,
) -> Result<Leaders> {
    let mut leaders = Leaders::default();
    let mut analyses = solver.analyses();
    let mut dom_info = analyses.get_analysis_mut::<DomInfo>(op, ctx)?;
    for region in op.deref(ctx).regions() {
        compute_region_leaders(ctx, region, solver, &mut dom_info, insertions, &mut leaders);
    }
    Ok(leaders)
}

fn compute_op_leaders(
    ctx: &Context,
    op: Ptr<Operation>,
    solver: &DataflowSolver,
    dom_info: &mut DomInfo,
    insertions: &IMap<ProgramPoint, SparseBitSet>,
    leaders: &mut Leaders,
) {
    let Some(block) = op.deref(ctx).get_parent_block() else {
        return;
    };

    let before = ProgramPoint::before_op(ctx, op);
    let after = ProgramPoint::after_op(ctx, op);
    let block_end = ProgramPoint::at_block_end(ctx, block);

    if let Some(insertions) = insertions.get(&before) {
        let after = ProgramPoint::at_block_end(ctx, op.deref(ctx).get_parent_block().unwrap());
        let leaders_after = leaders.leaders.entry(after).or_default();
        for insertion in insertions {
            let value = insertion as u64;
            leaders_after.insert(value, Leader::Insertion(before, value));
        }
    }

    // Region ops should snapshot the state before the region entry because it may be an insertion point
    if op.impls::<dyn RegionBranchOpInterface>(ctx) {
        let before = ProgramPoint::before_op(ctx, op);
        let after = ProgramPoint::at_block_end(ctx, op.deref(ctx).get_parent_block().unwrap());
        let leaders_after = leaders.leaders.entry(after).or_default().clone();
        let leaders_before = leaders.leaders.entry(before).or_default();
        leaders_before.extend(leaders_after.clone());
    }

    for region in op.deref(ctx).regions() {
        compute_region_leaders(ctx, region, solver, dom_info, insertions, leaders);
    }

    let avail_before = available_at(solver, before).unwrap();
    let avail_after = available_at(solver, after).unwrap();
    let new_values: SparseBitSet = avail_after.difference(&avail_before).into();

    let leaders_after = leaders.leaders.entry(block_end).or_default();

    for result in op.deref(ctx).results() {
        let value = number_of_value(solver, result).unwrap();
        if new_values.contains(value as usize) {
            leaders_after.insert(value, Leader::Material(result));
        }
    }

    for new_value in new_values.iter() {
        let new_value = new_value as u64;
        leaders_after
            .entry(new_value)
            .or_insert(Leader::RegionArg(op, RegionSuccessor::AfterOp));
    }
}

fn compute_region_leaders(
    ctx: &Context,
    region: Ptr<Region>,
    solver: &DataflowSolver,
    dom_info: &mut DomInfo,
    insertions: &IMap<ProgramPoint, SparseBitSet>,
    leaders: &mut Leaders,
) {
    let Some(entry) = region.entry_node(ctx) else {
        return;
    };

    let parent = region.deref(ctx).get_parent_op();
    let initial = match parent.deref(ctx).get_parent_block() {
        Some(block) if !parent.impls::<dyn IsolatedFromAboveInterface>(ctx) => {
            let point = ProgramPoint::at_block_end(ctx, block);
            leaders.leaders.entry(point).or_default().clone()
        }
        _ => Default::default(),
    };

    let point = ProgramPoint::at_block_end(ctx, entry);
    leaders.leaders.insert(point, initial);

    let mut worklist = VecDeque::new();
    worklist.push_back(entry);

    while let Some(block) = worklist.pop_front() {
        let start = ProgramPoint::at_block_start(ctx, block);
        let point = ProgramPoint::at_block_end(ctx, block);

        {
            let leaders = leaders.leaders.entry(point).or_default();
            for arg in block.deref(ctx).arguments() {
                let value = number_of_value(solver, arg).unwrap();
                leaders.entry(value).or_insert(Leader::Material(arg));
            }

            // Available values not transferred from a dominator need a block arg
            for value in available_at(solver, start).unwrap().iter() {
                let value = value as u64;
                if block.is_entry_block(ctx) {
                    leaders
                        .entry(value)
                        .or_insert(Leader::RegionArg(parent, region.into()));
                } else {
                    leaders.entry(value).or_insert(Leader::BlockArg(block));
                }
            }
        }

        for op in block.deref(ctx).iter(ctx) {
            compute_op_leaders(ctx, op, solver, dom_info, insertions, leaders);
        }

        {
            let leaders = leaders.leaders.entry(point).or_default();
            if let Some(insertions) = insertions.get(&point) {
                for insertion in insertions {
                    let value = insertion as u64;
                    leaders.insert(value, Leader::Insertion(point, value));
                }
            }
        }

        let dom_tree = dom_info.get_dom_tree(ctx, region);
        let leaders_at_end = leaders.leaders.entry(point).or_default().clone();
        for dominated in dom_tree.children(&block) {
            let after = ProgramPoint::at_block_end(ctx, dominated);
            leaders.leaders.insert(after, leaders_at_end.clone());
            worklist.push_back(dominated);
        }
    }
}

fn insert_expressions(
    ctx: &mut Context,
    solver: &DataflowSolver,
    mut insertions: IMap<ProgramPoint, SparseBitSet>,
    antileaders: &HashMap<u64, Ptr<Operation>>,
    leaders: &mut Leaders,
    rewriter: &mut impl Rewriter,
) -> IRStatus {
    let status = match insertions.is_empty() {
        true => IRStatus::Unchanged,
        false => IRStatus::Changed,
    };
    while !insertions.is_empty() {
        insertions.retain(|point, insertions| {
            if let Some(prev) = point.prev_op(ctx)
                && prev.impls::<dyn IsTerminatorInterface>(ctx)
            {
                rewriter.set_insertion_point_before_operation(prev);
            } else {
                rewriter.set_insertion_point((*point).into());
            }

            for expr in insertions.clone().into_iter().map(|it| it as u64) {
                let antileader = antileaders[&expr];
                if try_insert_expression(ctx, solver, *point, expr, antileader, leaders, rewriter) {
                    insertions.remove(expr as usize);
                }
            }
            !insertions.is_empty()
        });
    }
    status
}

fn try_insert_expression(
    ctx: &mut Context,
    solver: &DataflowSolver,
    point: ProgramPoint,
    expression: u64,
    antileader: Ptr<Operation>,
    leaders: &mut Leaders,
    rewriter: &mut impl Rewriter,
) -> bool {
    let mut operands = Vec::with_capacity(antileader.deref(ctx).get_num_operands());
    for operand in antileader.operands(ctx) {
        let expression = number_of_value(solver, operand).unwrap();
        let Some(leader) = leaders.try_materialize(ctx, &point, expression, operand.get_type(ctx))
        else {
            return false;
        };
        operands.push(leader);
    }
    let result_ty = antileader.result_types(ctx);
    let attributes = antileader.deref(ctx).attributes.clone();
    // This shouldn't be a trait in the first place but I can't be bothered to convince the Pliron
    // maintainer to make op info public right now, so just require this.
    let materializable = antileader.cast::<dyn MaterializableOp>(ctx).unwrap();
    let op = materializable.materialize(ctx, result_ty, operands, attributes);
    rewriter.append_operation(ctx, op);
    leaders
        .materialized
        .insert(Leader::Insertion(point, expression), op.result(ctx));
    true
}

fn replace_redundant_values(
    ctx: &mut Context,
    solver: &DataflowSolver,
    op: Ptr<Operation>,
    leaders: &mut Leaders,
    rewriter: &mut impl Rewriter,
) -> IRStatus {
    let mut status = IRStatus::Unchanged;
    visit_all_values_mut(
        ctx,
        &mut (solver, leaders, rewriter, &mut status),
        op,
        |ctx, (solver, leaders, rewriter, status), value| {
            let Some(block) = value.get_defining_block(ctx) else {
                return;
            };
            let block_end = ProgramPoint::at_block_end(ctx, block);
            let Some(expr) = number_of_value(solver, value) else {
                return;
            };
            let leader = leaders.materialize(ctx, &block_end, expr, value.get_type(ctx));
            if leader != value {
                rewriter.replace_value_uses_with(ctx, value, leader);
                **status |= IRStatus::Changed;
            }
        },
    );
    status
}

fn available_at(solver: &DataflowSolver, point: ProgramPoint) -> Option<SparseBitSet> {
    Some(
        solver
            .lookup_state::<Available>(point)?
            .deref()
            .value()
            .0
            .initialized()?
            .clone(),
    )
}

fn collect_antileaders(
    ctx: &Context,
    op: Ptr<Operation>,
    solver: &DataflowSolver,
    insertions: &IMap<ProgramPoint, SparseBitSet>,
) -> HashMap<u64, Ptr<Operation>> {
    let required_antileaders: SparseBitSet = hi_sparse_bitset::reduce(ops::Or, insertions.values())
        .map(Into::into)
        .unwrap_or_default();

    let mut antileaders = HashMap::new();
    walk_op(
        ctx,
        &mut (&mut antileaders, solver, &required_antileaders),
        &WALKCONFIG_PREORDER_REVERSE,
        op,
        |ctx, (antileaders, solver, required), node| {
            let IRNode::Operation(op) = node else {
                return;
            };
            if op.deref(ctx).num_regions() > 0 || op.deref(ctx).get_num_results() != 1 {
                return;
            }
            let Some(value) = number_of_value(solver, op.result(ctx)) else {
                return;
            };
            if required.contains(value as usize) && !antileaders.contains_key(&value) {
                antileaders.insert(value, op);
            }
        },
    );
    antileaders
}

pub(crate) fn number_of_value(solver: &DataflowSolver, value: Value) -> Option<u64> {
    let value = solver.lookup_state::<ValueNumberLattice>(value)?;
    value.deref().value().value()
}
