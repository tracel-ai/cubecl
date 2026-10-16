use core::fmt;

use alloc::{collections::VecDeque, format};
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::{BlockPtrExt, RegionPtrExt},
    interfaces::control_flow::RegionBranchOpInterface,
    prelude::*,
};
use derive_new::new;
use hi_sparse_bitset::ops;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    graph::{
        ControlFlowGraph, HasLabel, dominance::DomInfo,
        walkers::uninterruptible::immutable::walk_op,
    },
    linked_list::ContainsLinkedList,
    operation::OpDbg,
    printable::{self, Printable, indented_nl},
    region::Region,
    utils::table::{HMap, IMap},
};
use smallvec::smallvec;

use crate::{
    BitSetExt, SparseBitSet,
    analyses::dataflow_solver::{
        DataflowSolver, ProgramPoint, SmallPtrVec,
        pre::{
            AnticipatedMode, Available, InsertMode, compute_lcm_analyses,
            compute_pruned_insertion_points,
        },
        value_numbering::ValueNumberLattice,
    },
};

pub fn run_pre(
    ctx: &Context,
    op: Ptr<Operation>,
    antic_mode: AnticipatedMode,
    insert_mode: InsertMode,
) -> Result<IRStatus> {
    let mut solver = compute_lcm_analyses(ctx, op, antic_mode)?;
    let insertions = compute_pruned_insertion_points(ctx, op, &mut solver, insert_mode)?;

    let antileaders = collect_antileaders(ctx, op, &solver, &insertions);
    let leaders = compute_leaders(ctx, op, &solver, &insertions)?;

    std::println!("{}", leaders.disp(ctx));

    todo!()
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
    let after = ProgramPoint::at_block_end(ctx, block);

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

    let leaders_after = leaders.leaders.entry(after).or_default();

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
            .or_insert(Leader::RegionResult(op));
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
        let point = ProgramPoint::at_block_end(ctx, block);

        {
            let leaders = leaders.leaders.entry(point).or_default();
            for arg in block.deref(ctx).arguments() {
                let value = number_of_value(solver, arg).unwrap();
                leaders.entry(value).or_insert(Leader::Material(arg));
            }

            // Available values not transferred from a dominator need a block arg
            for value in available_at(solver, point).unwrap().iter() {
                let value = value as u64;
                leaders.entry(value).or_insert(Leader::BlockArg(block));
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

#[derive(Default, Debug)]
struct Leaders {
    leaders: HashMap<ProgramPoint, HashMap<u64, Leader>>,
    materialized: HashMap<Leader, Value>,
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
        f.write_str("\n}")
    }
}

#[derive(Clone, Copy, Debug)]
enum Leader {
    Material(Value),
    RegionResult(Ptr<Operation>),
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
            Leader::RegionResult(op) => write!(f, "RegionResult({})", OpDbg { op: *op, ctx }),
            Leader::BlockArg(block) => write!(f, "BlockArg({})", block.label(ctx)),
            Leader::Insertion(point, expr) => write!(f, "Insertion({}, e{expr})", point.disp(ctx)),
        }
    }
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

fn number_of_value(solver: &DataflowSolver, value: Value) -> Option<u64> {
    let value = solver.lookup_state::<ValueNumberLattice>(value)?;
    value.deref().value().value()
}
