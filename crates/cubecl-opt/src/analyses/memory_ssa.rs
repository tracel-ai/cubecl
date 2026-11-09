use core::{fmt, hash::Hash};

use alloc::{
    collections::VecDeque,
    format,
    string::{String, ToString},
};
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::{RegionPtrExt, UniquedExt},
    interfaces::{
        control_flow::CallableOpInterface,
        memory_slot::{
            MemorySSAContext, MemorySSARegionOpInterface, MemoryValue, RegionMemoryPhiInputs,
            RegionMemoryValue,
        },
        side_effects::{MemoryEffect, MemoryEffectsOp},
    },
    prelude::*,
};
use derive_more::From;
use derive_new::new;
use hashbrown::Equivalent;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    graph::{
        ControlFlowGraph, HasLabel, dominance::DomInfo,
        walkers::uninterruptible::immutable::walk_region,
    },
    linked_list::ContainsLinkedList,
    operation::OpDbg,
    printable::{self, Printable, fmt_indented_newline},
    region::Region,
    symbol_table::SymbolTableCollection,
    uniqued_any::Uniqued,
    utils::table::*,
    value::DefiningEntity,
};
use smallvec::SmallVec;

use crate::analyses::{
    alias_analysis::{AliasAnalysis, AliasAnalysisStack},
    dataflow_solver::{
        DataflowSolver, ProgramPoint, SolverConfig,
        dead_code::{DeadCodeAnalysis, PredecessorState},
        sccp::DummySparseConstantPropagation,
    },
    dominance::DomFrontierCalculator,
};

type MemoryEffectsMap = HMap<Ptr<Operation>, NodeMemoryEffects>;
type RegionMemoryEffectsMap = IMap<Ptr<Region>, MemoryEffectsMap>;
type DefiningBlocks = IMap<Ptr<Region>, SmallSet<Ptr<BasicBlock>, 16>>;

type RegionSet = SmallSet<Ptr<Region>, 32>;

pub type CallgraphPhiInputs = SmallMap<Ptr<Operation>, MemoryValue, 2>;

#[derive(Default)]
struct RegionMemoryInfo {
    has_memory_defs: bool,
}

#[derive(Default)]
struct MemorySSAInfo {
    merge_points: ISet<Ptr<BasicBlock>>,
    user_to_effects: RegionMemoryEffectsMap,
    regions_to_traverse: IMap<Ptr<Region>, RegionMemoryInfo>,
}

#[derive(new)]
struct MemoryOpAnalyzer<'a> {
    root: Ptr<Operation>,
    dom_info: &'a mut DomInfo,
    solver: &'a DataflowSolver,
    symbol_table: &'a mut SymbolTableCollection,
}

fn effect_is_def(effect: &MemoryEffect) -> bool {
    match effect {
        MemoryEffect::Read(_) | MemoryEffect::ReadAllInSpace(_) | MemoryEffect::ReadAll => false,
        MemoryEffect::Write(_)
        | MemoryEffect::WriteAllInSpace(_)
        | MemoryEffect::WriteAll
        | MemoryEffect::Opaque => true,
    }
}

impl MemoryOpAnalyzer<'_> {
    fn compute_memory_effects(
        &mut self,
        ctx: &Context,
        user_to_effects: &mut RegionMemoryEffectsMap,
        defining_blocks: &mut DefiningBlocks,
        regions_to_analyze: &mut IMap<Ptr<Region>, RegionMemoryInfo>,
    ) {
        let root_region = self.root.deref(ctx).get_region(0);

        let mut regions_with_direct_def = RegionSet::new();
        let mut regions_with_direct_use = RegionSet::new();

        #[derive(new)]
        struct State<'a> {
            user_to_effects: &'a mut RegionMemoryEffectsMap,
            defining_blocks: &'a mut DefiningBlocks,
            regions_with_direct_def: &'a mut RegionSet,
            regions_with_direct_use: &'a mut RegionSet,
            solver: &'a DataflowSolver,
            symbol_table: &'a mut SymbolTableCollection,
        }

        walk_region(
            ctx,
            &mut State::new(
                user_to_effects,
                defining_blocks,
                &mut regions_with_direct_def,
                &mut regions_with_direct_use,
                self.solver,
                self.symbol_table,
            ),
            &WALKCONFIG_ANY,
            root_region,
            |ctx, state, node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                let block = op.deref(ctx).get_parent_block().unwrap();
                let region = op.deref(ctx).get_parent_region(ctx).unwrap();
                if op.deref(ctx).num_regions() > 0 {
                    // Region ops are processed separately
                } else if let Some(call) = op.cast::<dyn CallOpInterface>(ctx)
                    && let Some(callable) = resolve_callable(state.symbol_table, ctx, &*call)
                    && let Some(callable) = callable.cast::<dyn CallableOpInterface>(ctx)
                {
                    let point = ProgramPoint::after_op(ctx, op);
                    let region_effects = state.user_to_effects.entry(region).or_default();
                    let preds = state.solver.lookup_state::<PredecessorState>(point);
                    if preds.is_none_or(|preds| !preds.deref().all_predecessors_known())
                        || callable.callable_region(ctx).is_none()
                    {
                        region_effects.insert(op, NodeMemoryEffects::Opaque);
                    }
                    state.regions_with_direct_use.insert(region);
                    state.regions_with_direct_def.insert(region);
                    let def_blocks = state.defining_blocks.entry(region).or_default();
                    def_blocks.insert(block);
                } else if let Some(mem_effects) = op_cast::<dyn MemoryEffectsOp>(&*op.dyn_op(ctx)) {
                    let effects = mem_effects.memory_effects(ctx);
                    if effects.is_empty() {
                        return;
                    }
                    state.regions_with_direct_use.insert(region);
                    if effects.iter().any(effect_is_def) {
                        state.regions_with_direct_def.insert(region);
                        let def_blocks = state.defining_blocks.entry(region).or_default();
                        def_blocks.insert(block);
                    }
                    let region_effects = state.user_to_effects.entry(region).or_default();
                    region_effects.insert(op, NodeMemoryEffects::from_op(ctx, op));
                } else {
                    let region_effects = state.user_to_effects.entry(region).or_default();
                    region_effects.insert(op, NodeMemoryEffects::Opaque);
                    state.regions_with_direct_use.insert(region);
                    state.regions_with_direct_def.insert(region);
                    let def_blocks = state.defining_blocks.entry(region).or_default();
                    def_blocks.insert(block);
                }
            },
        );

        let mut visit_regions = |regions_to_propagate_from: &mut VecDeque<Ptr<Region>>,
                                 has_memory_defs: bool| {
            while let Some(region) = regions_to_propagate_from.pop_back() {
                if region == root_region || regions_to_analyze.contains_key(&region) {
                    continue;
                }

                regions_to_analyze.insert(region, RegionMemoryInfo { has_memory_defs });

                regions_to_propagate_from
                    .push_back(region.deref(ctx).get_parent_region(ctx).unwrap());
            }
        };

        let mut regions_to_propagate_from = VecDeque::new();
        regions_to_propagate_from.extend(regions_with_direct_def);
        visit_regions(&mut regions_to_propagate_from, true);

        regions_to_propagate_from.clear();
        regions_to_propagate_from.extend(regions_with_direct_use);
        visit_regions(&mut regions_to_propagate_from, false);
    }

    fn compute_merge_points(
        &mut self,
        ctx: &Context,
        region: Ptr<Region>,
        defining_blocks: &SmallSet<Ptr<BasicBlock>, 16>,
        merge_points: &mut ISet<Ptr<BasicBlock>>,
    ) {
        if region.deref(ctx).iter(ctx).count() <= 1 {
            return;
        }

        let dom_tree = self.dom_info.get_dom_tree(ctx, region);
        let frontiers =
            DomFrontierCalculator::new(ctx, &region, dom_tree, defining_blocks.iter().copied());

        merge_points.extend(frontiers.compute());
    }

    fn compute_info(&mut self, ctx: &Context) -> MemorySSAInfo {
        let mut info = MemorySSAInfo::default();

        let mut defining_blocks = IMap::<_, SmallSet<_, 16>>::default();
        self.compute_memory_effects(
            ctx,
            &mut info.user_to_effects,
            &mut defining_blocks,
            &mut info.regions_to_traverse,
        );

        for (region, region_info) in info.regions_to_traverse.iter() {
            if region_info.has_memory_defs {
                let parent_block = region.deref(ctx).get_parent_block(ctx).unwrap();
                let parent_region = region.deref(ctx).get_parent_region(ctx).unwrap();
                defining_blocks
                    .entry(parent_region)
                    .or_default()
                    .insert(parent_block);
            }
        }

        for (&region, def_blocks) in defining_blocks.iter() {
            self.compute_merge_points(ctx, region, def_blocks, &mut info.merge_points);
        }

        info
    }
}

fn resolve_callable(
    symbol_table: &mut SymbolTableCollection,
    ctx: &Context,
    call: &dyn CallOpInterface,
) -> Option<Ptr<Operation>> {
    match call.callee(ctx) {
        CallOpCallable::Direct(symbol) => symbol_table
            .lookup_symbol_in_nearest_table(ctx, call.get_operation(), &symbol)
            .map(|it| it.get_operation()),
        CallOpCallable::Indirect(value) => value.defining_op(),
    }
}

#[derive(Clone, Copy)]
pub enum NodeMemoryEffects {
    Opaque,
    Op(TraitOp<dyn MemoryEffectsOp>),
}

impl NodeMemoryEffects {
    fn from_op(ctx: &Context, op: Ptr<Operation>) -> NodeMemoryEffects {
        NodeMemoryEffects::Op(TraitOp::try_from_op(op, ctx).unwrap())
    }

    pub fn effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        match self {
            NodeMemoryEffects::Opaque => vec![MemoryEffect::Opaque],
            NodeMemoryEffects::Op(trait_op_ptr) => trait_op_ptr.memory_effects(ctx),
        }
    }
}

#[derive(new)]
pub struct MemoryDef {
    pub input: MemoryValue,
    pub effects: NodeMemoryEffects,
    pub result: MemoryValue,
}

impl MemoryDef {
    pub fn effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        self.effects.effects(ctx)
    }
}

impl Printable for MemoryDef {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryDef { input, result, .. } = self;
        write!(
            f,
            "{result} = MemoryDef({input}, {})",
            print_effects(ctx, &self.effects)
        )
    }
}

#[derive(new, Clone)]
pub struct MemoryUse {
    pub input: MemoryValue,
    pub effects: NodeMemoryEffects,
    #[new(default)]
    pub is_optimized: bool,
}

impl MemoryUse {
    pub fn effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        self.effects.effects(ctx)
    }
}

impl Printable for MemoryUse {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryUse { input, effects, .. } = self;
        write!(f, "MemoryUse({input}, {})", print_effects(ctx, effects))
    }
}

#[derive(new)]
pub struct MemoryPhi {
    pub inputs: SmallMap<Ptr<BasicBlock>, MemoryValue, 4>,
    pub result: MemoryValue,
}

impl Printable for MemoryPhi {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryPhi { inputs, result } = self;
        let mut inputs = inputs
            .iter()
            .map(|(block, value)| format!("{{{} -> {value}}}", block.label(ctx)));
        write!(f, "{result} = MemoryPhi({})", inputs.join(", "))
    }
}

#[derive(new)]
pub struct MemoryRegionPhi {
    pub inputs: RegionMemoryPhiInputs,
    pub result: MemoryValue,
}

impl Printable for MemoryRegionPhi {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryRegionPhi { inputs, result } = self;
        let mut inputs = inputs
            .iter()
            .map(|(pred, value)| format!("{{{} -> {value}}}", pred.disp(ctx)));
        write!(f, "{result} = MemoryRegionPhi({})", inputs.join(", "))
    }
}

#[derive(new)]
pub struct MemoryCallPhi {
    pub inputs: CallgraphPhiInputs,
    pub result: MemoryValue,
}

impl Printable for MemoryCallPhi {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryCallPhi { inputs, result } = self;
        let mut inputs = inputs
            .iter()
            .map(|(pred, value)| format!("{{{} -> {value}}}", OpDbg { op: *pred, ctx }));
        write!(f, "{result} = MemoryCallPhi({})", inputs.join(", "))
    }
}

#[derive(new)]
pub struct MemoryCalleePhi {
    pub inputs: CallgraphPhiInputs,
    pub result: MemoryValue,
}

impl Printable for MemoryCalleePhi {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MemoryCalleePhi { inputs, result } = self;
        let mut inputs = inputs
            .iter()
            .map(|(pred, value)| format!("{{{} -> {value}}}", OpDbg { op: *pred, ctx }));
        write!(f, "{result} = MemoryCalleePhi({})", inputs.join(", "))
    }
}

#[derive(From)]
pub enum MemorySSANode {
    Def(MemoryDef),
    Use(MemoryUse),
    Phi(MemoryPhi),
    RegionPhi(MemoryRegionPhi),
    CallPhi(MemoryCallPhi),
    CalleePhi(MemoryCalleePhi),
}

impl MemorySSANode {
    pub fn input(&self) -> Option<MemoryValue> {
        match self {
            MemorySSANode::Def(def) => Some(def.input),
            MemorySSANode::Use(r#use) => Some(r#use.input),
            MemorySSANode::Phi(_)
            | MemorySSANode::RegionPhi(_)
            | MemorySSANode::CallPhi(_)
            | MemorySSANode::CalleePhi(_) => None,
        }
    }
}

impl Printable for MemorySSANode {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            MemorySSANode::Def(def) => def.fmt(ctx, state, f),
            MemorySSANode::Use(r#use) => r#use.fmt(ctx, state, f),
            MemorySSANode::Phi(phi) => phi.fmt(ctx, state, f),
            MemorySSANode::RegionPhi(phi) => phi.fmt(ctx, state, f),
            MemorySSANode::CallPhi(phi) => phi.fmt(ctx, state, f),
            MemorySSANode::CalleePhi(phi) => phi.fmt(ctx, state, f),
        }
    }
}

fn print_effects(ctx: &Context, effects: &NodeMemoryEffects) -> String {
    let effects = effects.effects(ctx);
    effects.iter().map(|it| it.disp(ctx).to_string()).join(", ")
}

struct MemorySSAGraphBuilder<'a> {
    solver: &'a DataflowSolver,
    live_on_entry: HMap<Ptr<Region>, MemoryValue>,

    ctx: &'a mut MemorySSAContext,
    reaching_defs: HashMap<Ptr<Operation>, MemoryValue>,
    reaching_at_block_end: HMap<Ptr<BasicBlock>, MemoryValue>,
    reaching_at_region_entry: HMap<Ptr<Region>, MemoryValue>,

    nodes: HashMap<DefiningEntity, MemorySSANode>,
    pending_phis: ISet<(Ptr<BasicBlock>, MemoryValue)>,
    pending_call_phis: ISet<(Ptr<Operation>, MemoryValue)>,
    pending_callee_phis: ISet<(Ptr<Operation>, MemoryValue)>,

    info: MemorySSAInfo,

    visited_blocks: HSet<Ptr<BasicBlock>>,
}

impl<'a> MemorySSAGraphBuilder<'a> {
    fn new(ctx: &'a mut MemorySSAContext, solver: &'a DataflowSolver) -> Self {
        Self {
            solver,
            live_on_entry: Default::default(),
            reaching_defs: Default::default(),
            reaching_at_block_end: Default::default(),
            reaching_at_region_entry: Default::default(),
            nodes: Default::default(),
            pending_phis: Default::default(),
            pending_call_phis: Default::default(),
            pending_callee_phis: Default::default(),
            info: Default::default(),
            visited_blocks: Default::default(),
            ctx,
        }
    }

    fn analyze_block(
        &mut self,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        mut reaching_def: MemoryValue,
        dom_info: &mut DomInfo,
    ) -> MemoryValue {
        if self.visited_blocks.contains(&block) {
            panic!("analyze_block was called twice on one block");
        }
        self.visited_blocks.insert(block);

        let block_ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in block_ops {
            let parent_region = op.deref(ctx).get_parent_region(ctx).unwrap();
            let region_effects = self.info.user_to_effects.entry(parent_region).or_default();
            if let Some(effects) = region_effects.remove(&op) {
                self.reaching_defs.insert(op, reaching_def);
                if effects.effects(ctx).iter().any(effect_is_def) {
                    let new_res = self.ctx.new_value_at_op(op);
                    let node = MemoryDef::new(reaching_def, effects, new_res);
                    self.nodes.insert(DefiningEntity::Op(op), node.into());
                    reaching_def = new_res;
                } else {
                    let effects = NodeMemoryEffects::from_op(ctx, op);
                    let node = MemoryUse::new(reaching_def, effects);
                    self.nodes.insert(DefiningEntity::Op(op), node.into());
                }
            } else if op.impls::<dyn CallOpInterface>(ctx) {
                self.reaching_defs.insert(op, reaching_def);
                let phi_result = self.ctx.new_value_at_op(op);
                self.pending_call_phis.insert((op, phi_result));
                reaching_def = phi_result;
            }

            let op_obj = op.dyn_op(ctx);
            if op.deref(ctx).num_regions() > 0 {
                let mut needs_analysis = false;
                let mut has_memory_defs = false;
                for region in op.regions(ctx) {
                    let Some(region_info) = self.info.regions_to_traverse.get(&region) else {
                        continue;
                    };
                    needs_analysis = true;
                    if !region_info.has_memory_defs {
                        continue;
                    }
                    has_memory_defs = true;
                    break;
                }

                if needs_analysis {
                    if let Some(memory_region) = op_cast::<dyn MemorySSARegionOpInterface>(&*op_obj)
                    {
                        let mut regions_to_process = SmallMap::new();

                        memory_region.setup_memory_ssa(
                            ctx,
                            self.ctx,
                            reaching_def,
                            has_memory_defs,
                            &mut regions_to_process,
                        );

                        for (region, reaching_def) in regions_to_process {
                            self.reaching_at_region_entry.insert(region, reaching_def);
                            if !self.info.regions_to_traverse.contains_key(&region) {
                                continue;
                            }
                            self.analyze_region(ctx, region, reaching_def, dom_info);
                        }

                        reaching_def =
                            self.finalize_region(ctx, memory_region, reaching_def, has_memory_defs);
                    } else {
                        for region in op.regions(ctx) {
                            let Some(entry) = region.deref(ctx).get_entry_block() else {
                                continue;
                            };
                            let reaching_def =
                                self.opaque_at(reaching_def, DefiningEntity::Block(entry));
                            self.reaching_at_region_entry.insert(region, reaching_def);
                            if !self.info.regions_to_traverse.contains_key(&region) {
                                continue;
                            }
                            self.analyze_region(ctx, region, reaching_def, dom_info);
                        }

                        reaching_def = self.opaque_at(reaching_def, DefiningEntity::Op(op));
                    }
                }
            }
        }

        self.reaching_at_block_end.insert(block, reaching_def);
        if let Some(term) = block.deref(ctx).get_terminator(ctx) {
            self.reaching_defs.insert(term, reaching_def);
        }

        reaching_def
    }

    fn opaque_at(&mut self, input: MemoryValue, defining: DefiningEntity) -> MemoryValue {
        let new_reaching = match defining {
            DefiningEntity::Op(op) => self.ctx.new_value_at_op(op),
            DefiningEntity::Block(block) => self.ctx.new_value_in_block(block),
        };
        let clobber = MemoryDef::new(input, NodeMemoryEffects::Opaque, new_reaching);
        self.nodes.insert(defining, clobber.into());
        new_reaching
    }

    fn finalize_region(
        &mut self,
        ctx: &Context,
        memory_region: &dyn MemorySSARegionOpInterface,
        reaching_def: MemoryValue,
        has_memory_defs: bool,
    ) -> MemoryValue {
        let op = memory_region.get_operation();
        let mut region_phis = SmallMap::new();
        let result = memory_region.finalize_memory_ssa(
            ctx,
            self.ctx,
            reaching_def,
            has_memory_defs,
            &self.reaching_at_region_entry,
            &self.reaching_at_block_end,
            &mut region_phis,
        );

        // Attach phi nodes to the region entries that need it, the region version of `link_merge_points`
        for (region, inputs) in region_phis {
            let entry = region.deref(ctx).get_entry_block().unwrap();
            let result = *self.reaching_at_region_entry.get(&region).unwrap();
            let node = MemoryRegionPhi::new(inputs, result);
            self.nodes.insert(DefiningEntity::Block(entry), node.into());
        }

        match result {
            RegionMemoryValue::Forward(value) => value,
            RegionMemoryValue::RegionPhi(inputs) => {
                let result = self.ctx.new_value_at_op(op);
                let node = MemoryRegionPhi::new(inputs, result);
                self.nodes.insert(DefiningEntity::Op(op), node.into());
                result
            }
        }
    }

    fn analyze_region(
        &mut self,
        ctx: &Context,
        region: Ptr<Region>,
        reaching_def: MemoryValue,
        dom_info: &mut DomInfo,
    ) {
        self.live_on_entry.insert(region, reaching_def);
        if region.is_empty(ctx) {
            return;
        }

        let entry = region.entry_node(ctx).unwrap();

        let mut dfs_stack = VecDeque::new();
        dfs_stack.push_back((entry, reaching_def));

        while let Some((block, mut reaching_def)) = dfs_stack.pop_back() {
            if self.info.merge_points.contains(&block) {
                let phi_result = self.ctx.new_value_in_block(block);
                self.pending_phis.insert((block, phi_result));
                reaching_def = phi_result;
            }

            reaching_def = self.analyze_block(ctx, block, reaching_def, dom_info);
            let dom_tree = dom_info.get_dom_tree(ctx, region);

            for child in dom_tree.children(&block) {
                dfs_stack.push_back((child, reaching_def));
            }
        }
    }

    /// Resolve the incoming values for memory-state phis at CFG merge points after all intra-block
    /// reaching definitions have been resolved.
    fn link_merge_points(&mut self, ctx: &Context) {
        while let Some((merge_point, result)) = self.pending_phis.pop() {
            let mut inputs = SmallMap::new();
            for r#use in merge_point.uses(ctx) {
                let user_block = r#use.user_op().deref(ctx).get_parent_block().unwrap();
                let user_region = user_block.deref(ctx).get_parent_region().unwrap();
                let reaching_def = self.reaching_at_block_end.get(&user_block).copied();
                let reaching_def = reaching_def.unwrap_or(self.live_on_entry[&user_region]);

                inputs.insert(user_block, reaching_def);
            }

            let node = MemoryPhi::new(inputs, result);
            self.nodes
                .insert(DefiningEntity::Block(merge_point), node.into());
        }
    }

    fn analyze_callable(
        &mut self,
        ctx: &Context,
        callable_region: Ptr<Region>,
        dom_info: &mut DomInfo,
        info: MemorySSAInfo,
    ) {
        self.info = info;

        let callable_op = callable_region.deref(ctx).get_parent_op();
        let callable_point = ProgramPoint::after_op(ctx, callable_op);
        let predecessors = self.solver.lookup_state::<PredecessorState>(callable_point);

        let live_on_entry = if let Some(predecessors) = predecessors
            && predecessors.deref().all_predecessors_known()
        {
            let phi_result = self.ctx.new_value_at_op(callable_op);
            self.pending_callee_phis.insert((callable_op, phi_result));
            phi_result
        } else {
            self.ctx.new_value_at_entry()
        };

        // Build MemorySSA recursively through nested regions. The initial reaching
        // memory definition is the live-on-entry value.
        //
        // Nested region operations are finalized as their surrounding operations are
        // processed, after the region bodies have established their reaching memory
        // definitions.
        self.analyze_region(ctx, callable_region, live_on_entry, dom_info);

        // Finally, connect merge points to their predecessor's reaching definitions.
        self.link_merge_points(ctx);
    }

    fn link_call_predecessors(&mut self, ctx: &Context) {
        for (merge_point, result) in self.pending_call_phis.drain(..) {
            let mut inputs = SmallMap::new();
            let point = ProgramPoint::after_op(ctx, merge_point);
            let predecessors = self.solver.lookup_state::<PredecessorState>(point).unwrap();

            for predecessor in predecessors.deref().known_predecessors() {
                inputs.insert(*predecessor, self.reaching_defs[predecessor]);
            }

            let node = MemoryCallPhi::new(inputs, result);
            self.nodes
                .insert(DefiningEntity::Op(merge_point), node.into());
        }

        for (merge_point, result) in self.pending_callee_phis.drain(..) {
            let mut inputs = SmallMap::new();
            let point = ProgramPoint::after_op(ctx, merge_point);
            let predecessors = self.solver.lookup_state::<PredecessorState>(point).unwrap();

            for predecessor in predecessors.deref().known_predecessors() {
                inputs.insert(*predecessor, self.reaching_defs[predecessor]);
            }

            let node = MemoryCalleePhi::new(inputs, result);
            self.nodes
                .insert(DefiningEntity::Op(merge_point), node.into());
        }
    }
}

pub struct MemorySSA {
    root: Ptr<Operation>,
    nodes: HashMap<DefiningEntity, MemorySSANode>,
    analysis_stack: AliasAnalysisStack,
}

impl MemorySSA {
    /// Empty analysis, can be used as a dummy when `MemorySSA` is optional.
    pub fn empty(root: Ptr<Operation>) -> Self {
        Self {
            root,
            nodes: Default::default(),
            analysis_stack: AliasAnalysisStack::default(),
        }
    }
}

impl Printable for MemorySSA {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str("MemorySSA {")?;
        {
            let _indent = state.indent();
            for region in self.root.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    print_block(ctx, state, &self.nodes, block, f)?;
                }
            }
        }
        f.write_str("\n}")
    }
}

fn print_block(
    ctx: &Context,
    state: &printable::State,
    nodes: &HashMap<DefiningEntity, MemorySSANode>,
    block: Ptr<BasicBlock>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    if let Some(node) = nodes.get(&DefiningEntity::Block(block)) {
        fmt_indented_newline(state, f)?;
        write!(f, "; {}", node.disp(ctx))?;
    }

    fmt_indented_newline(state, f)?;
    write!(f, "{}:", block.label(ctx))?;
    {
        let _indent = state.indent();
        for op in block.deref(ctx).iter(ctx) {
            print_op(ctx, state, nodes, op, f)?;
        }
    }
    writeln!(f)
}

fn print_op(
    ctx: &Context,
    state: &printable::State,
    nodes: &HashMap<DefiningEntity, MemorySSANode>,
    op: Ptr<Operation>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    if let Some(node) = nodes.get(&DefiningEntity::Op(op)) {
        fmt_indented_newline(state, f)?;
        write!(f, "; {}", node.disp(ctx))?;
    }

    fmt_indented_newline(state, f)?;
    write!(f, "{};", OpDbg { op, ctx }.to_string().trim())?;
    if op.deref(ctx).num_regions() > 0 {
        let _indent = state.indent();
        for region in op.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                print_block(ctx, state, nodes, block, f)?;
            }
        }
    }
    Ok(())
}

impl MemorySSA {
    pub fn node_for_op(&self, op: Ptr<Operation>) -> Option<&MemorySSANode> {
        self.nodes.get(&DefiningEntity::Op(op))
    }

    pub fn node_for_block(&self, op: Ptr<BasicBlock>) -> Option<&MemorySSANode> {
        self.nodes.get(&DefiningEntity::Block(op))
    }

    pub fn node_for_value(&self, value: MemoryValue) -> Option<&MemorySSANode> {
        self.nodes.get(&value.defining_entity()?)
    }

    pub fn walker<'a>(
        &'a self,
        ctx: &'a Context,
        value: MemoryValue,
        effects: &'a [MemoryEffect],
    ) -> MemorySSAWalker<'a> {
        MemorySSAWalker {
            is_interprocedural: true,
            stop_at_phi: false,
            effects,
            start_value: value,
            ctx,
            memory_ssa: self,
        }
    }

    pub fn set_alias_analysis_stack(&mut self, stack: AliasAnalysisStack) {
        self.analysis_stack = stack;
    }

    pub fn add_alias_analysis(&mut self, analysis: impl AliasAnalysis + 'static) {
        self.analysis_stack.add_analysis(analysis);
    }

    pub fn ensure_optimized_uses(&mut self, ctx: &Context) {
        let mut updated_nodes = vec![];
        for (defining, node) in self.nodes.iter() {
            let MemorySSANode::Use(r#use) = node else {
                continue;
            };
            if r#use.is_optimized {
                continue;
            }
            let effects = r#use.effects(ctx);
            let mut walker = self.walker(ctx, r#use.input, &effects);
            let clobbering_access = walker.next_clobbering_def();
            let mut new_use = MemoryUse::new(clobbering_access, r#use.effects);
            new_use.is_optimized = true;

            updated_nodes.push((*defining, new_use));
        }
        for (defining, node) in updated_nodes {
            self.nodes.insert(defining, node.into());
        }
    }

    pub fn optimized_use(&mut self, ctx: &Context, op: Ptr<Operation>) -> Option<MemoryValue> {
        let Some(MemorySSANode::Use(r#use)) = self.node_for_op(op) else {
            return None;
        };
        if r#use.is_optimized {
            return Some(r#use.input);
        }
        let effects = r#use.effects(ctx);
        let mut walker = self.walker(ctx, r#use.input, &effects);
        let clobbering_access = walker.next_clobbering_def();
        let mut new_use = MemoryUse::new(clobbering_access, r#use.effects);
        new_use.is_optimized = true;
        self.nodes.insert(DefiningEntity::Op(op), new_use.into());
        Some(clobbering_access)
    }

    pub fn op_memory_def(&self, op: Ptr<Operation>) -> Option<MemoryValue> {
        self.memory_def(DefiningEntity::Op(op))
    }

    pub fn block_memory_def(&self, block: Ptr<BasicBlock>) -> Option<MemoryValue> {
        self.memory_def(DefiningEntity::Block(block))
    }

    pub fn memory_def(&self, defining: DefiningEntity) -> Option<MemoryValue> {
        let node = self.nodes.get(&defining)?;
        match node {
            MemorySSANode::Def(def) => Some(def.result),
            MemorySSANode::Use(_) => None,
            MemorySSANode::Phi(phi) => Some(phi.result),
            MemorySSANode::RegionPhi(phi) => Some(phi.result),
            MemorySSANode::CallPhi(phi) => Some(phi.result),
            MemorySSANode::CalleePhi(phi) => Some(phi.result),
        }
    }
}

pub struct MemorySSAWalker<'a> {
    is_interprocedural: bool,
    stop_at_phi: bool,
    effects: &'a [MemoryEffect],
    start_value: MemoryValue,
    ctx: &'a Context,
    memory_ssa: &'a MemorySSA,
}

// Modifying the call stack is the cold path, copying is the hot path. So unique it to make copying cheap.
type CallStack = Uniqued<SmallVec<[Ptr<Operation>; 8]>>;

#[derive(Default, Clone, Debug)]
struct VisitedSet {
    set: SmallVec<[(MemoryValue, CallStack); 4]>,
}

impl VisitedSet {
    fn insert(&mut self, value: MemoryValue, stack: CallStack) {
        if !self.contains(&value, &stack) {
            self.set.push((value, stack));
        }
    }

    fn contains(&self, value: &MemoryValue, stack: &CallStack) -> bool {
        self.set.iter().any(|(v, s)| v == value && s == stack)
    }
}

#[derive(Hash)]
struct ExploredKey<'a>(&'a MemoryValue, &'a CallStack);

impl Equivalent<(MemoryValue, CallStack)> for ExploredKey<'_> {
    fn equivalent(&self, key: &(MemoryValue, CallStack)) -> bool {
        &key.0 == self.0 && &key.1 == self.1
    }
}

#[derive(Default, Debug)]
struct ExploredMap {
    map: HashMap<(MemoryValue, CallStack), ExploredResult>,
}

impl ExploredMap {
    fn insert(&mut self, value: MemoryValue, stack: CallStack, result: ExploredResult) {
        self.map.insert((value, stack), result);
    }

    fn get(&self, value: &MemoryValue, stack: &CallStack) -> Option<&ExploredResult> {
        self.map.get(&ExploredKey(value, stack))
    }
}

#[derive(Debug, Clone)]
enum TraversalResult {
    Value(MemoryValue, CallStack),
    Cycle,
}

#[derive(Debug, Clone)]
enum ExploredResult {
    Advance(MemoryValue, CallStack),
    Clobber(MemoryValue),
    Cycle,
}

/// One walk on the explicit stack of [`MemorySSAWalker::next_clobbering_def_impl`].
#[derive(Debug)]
struct WalkFrame {
    state: WalkState,
    visited: VisitedSet,
    call_stack: CallStack,
}

/// Where a walk stands.
#[derive(Debug)]
enum WalkState {
    /// Walking up from this value.
    Walking(MemoryValue),
    /// Stopped at a phi, waiting on its inputs' sub-walks.
    AtPhi(PhiWalk),
    /// Stopped at a call, waiting on its callee's sub-walks.
    AtCall(PhiWalk),
    /// Done, with this result.
    Settled(TraversalResult),
}

impl WalkFrame {
    fn new(start: MemoryValue, visited: VisitedSet, call_stack: CallStack) -> Self {
        Self {
            state: WalkState::Walking(start),
            visited,
            call_stack,
        }
    }

    /// Takes the result of the sub-walk from the phi's latest input. A cycle is dropped, since
    /// cycles only happen on paths without clobbers; two inputs reaching different clobbers make
    /// the phi itself the clobber.
    fn take_input_result(&mut self, result: TraversalResult, explored: &mut ExploredMap) {
        let (WalkState::AtPhi(phi) | WalkState::AtCall(phi)) = &mut self.state else {
            unreachable!("a sub-walk only runs for a phi");
        };
        let TraversalResult::Value(value, call_stack) = result else {
            return;
        };
        match &phi.agreed {
            // Unwind causes the traversal result to have the call itself as the result, which would
            // lead to self-referential phis if not checked. A self-referential phi should be treated
            // as if there is no agreement.
            None if value != phi.value || call_stack != self.call_stack => {
                phi.agreed = Some((value, call_stack))
            }
            Some(agreed) if agreed == &(value, call_stack) => {}
            _ => {
                explored.insert(
                    phi.value,
                    self.call_stack,
                    ExploredResult::Clobber(phi.value),
                );
                self.state = WalkState::Settled(TraversalResult::Value(phi.value, self.call_stack));
            }
        }
    }
}

/// A phi being resolved: which of its inputs has been walked, and the clobber the walked ones
/// agree on.
#[derive(Debug)]
struct PhiWalk {
    value: MemoryValue,
    next: usize,
    agreed: Option<(MemoryValue, CallStack)>,
}

impl PhiWalk {
    fn new(value: MemoryValue) -> Self {
        Self {
            value,
            next: 0,
            agreed: None,
        }
    }

    /// The next input not on the current path, read off the phi's node.
    fn next_unvisited(
        &mut self,
        memory_ssa: &MemorySSA,
        visited: &VisitedSet,
        call_stack: &CallStack,
    ) -> Option<MemoryValue> {
        while let Some(input) = self.next_input(memory_ssa) {
            self.next += 1;
            if !visited.contains(&input, call_stack) {
                return Some(input);
            }
        }
        None
    }

    fn next_input(&self, memory_ssa: &MemorySSA) -> Option<MemoryValue> {
        match memory_ssa.node_for_value(self.value)? {
            MemorySSANode::Phi(phi) => phi.inputs.iter().nth(self.next).map(|(_, value)| *value),
            MemorySSANode::RegionPhi(phi) => {
                phi.inputs.iter().nth(self.next).map(|(_, value)| *value)
            }
            MemorySSANode::CallPhi(phi) => {
                phi.inputs.iter().nth(self.next).map(|(_, value)| *value)
            }
            // Callee phi is not a real phi - it gets narrowed to a single source based on the call stack
            MemorySSANode::Def(_) | MemorySSANode::Use(_) | MemorySSANode::CalleePhi(_) => None,
        }
    }

    /// Every input walked: the clobber they agree on, or a cycle if none reached one.
    fn resolve(&self, call_stack: &mut CallStack, explored: &mut ExploredMap) -> WalkState {
        match &self.agreed {
            Some((value, agreed_call_stack)) => {
                explored.insert(
                    self.value,
                    *call_stack,
                    ExploredResult::Advance(*value, *agreed_call_stack),
                );

                *call_stack = *agreed_call_stack;
                WalkState::Walking(*value)
            }
            None => {
                explored.insert(self.value, *call_stack, ExploredResult::Cycle);
                WalkState::Settled(TraversalResult::Cycle)
            }
        }
    }
}

/// What a frame needs next from the stack driving it.
enum WalkStep {
    /// A sub-walk from this phi input.
    Descend(MemoryValue),
    /// A sub-walk from this call
    EnterCall(MemoryValue, Ptr<Operation>),
    /// Nothing: the frame is done.
    Return(TraversalResult),
}

impl<'a> MemorySSAWalker<'a> {
    pub fn stop_at_phi(&mut self) {
        self.stop_at_phi = true;
    }

    pub fn next_clobbering_def(&mut self) -> MemoryValue {
        let call_stack = Uniqued::new(self.ctx, Default::default());
        match self.next_clobbering_def_impl(
            &mut ExploredMap::default(),
            call_stack,
            VisitedSet::default(),
        ) {
            TraversalResult::Value(value, _) => value,
            TraversalResult::Cycle => self.start_value,
        }
    }

    /// Walks up from `start_value` to the access clobbering `effects`.
    ///
    /// A phi forks the walk into one sub-walk per input. The sub-walks run on an explicit stack
    /// rather than the call stack: a kernel with many rolled loops chains phis deep enough to
    /// overflow the thread compiling it.
    fn next_clobbering_def_impl(
        &mut self,
        explored: &mut ExploredMap,
        call_stack: CallStack,
        visited: VisitedSet,
    ) -> TraversalResult {
        let mut stack = vec![WalkFrame::new(self.start_value, visited, call_stack)];
        loop {
            let frame = stack.last_mut().expect("the root frame is popped last");
            match self.advance(frame, explored) {
                WalkStep::EnterCall(value, call) => {
                    let visited = frame.visited.clone();
                    let call_stack = frame.call_stack.map(self.ctx, |mut call_stack| {
                        call_stack.push(call);
                        call_stack
                    });
                    stack.push(WalkFrame::new(value, visited, call_stack));
                }
                WalkStep::Descend(value) => {
                    let visited = frame.visited.clone();
                    let call_stack = frame.call_stack;
                    stack.push(WalkFrame::new(value, visited, call_stack));
                }
                WalkStep::Return(result) => {
                    stack.pop();
                    match stack.last_mut() {
                        Some(parent) => parent.take_input_result(result, explored),
                        None => return result,
                    }
                }
            }
        }
    }

    /// Runs `frame` until it finishes or needs the result of a phi input's sub-walk.
    fn advance(&mut self, frame: &mut WalkFrame, explored: &mut ExploredMap) -> WalkStep {
        loop {
            match &mut frame.state {
                WalkState::Walking(value) => {
                    frame.state =
                        self.step(*value, &mut frame.visited, &mut frame.call_stack, explored);
                }
                WalkState::AtCall(phi) => {
                    let call_op = phi.value.defining_op().expect("Calls are always ops");
                    match phi.next_unvisited(self.memory_ssa, &frame.visited, &frame.call_stack) {
                        Some(input) => return WalkStep::EnterCall(input, call_op),
                        None => frame.state = phi.resolve(&mut frame.call_stack, explored),
                    }
                }
                WalkState::AtPhi(phi) => {
                    match phi.next_unvisited(self.memory_ssa, &frame.visited, &frame.call_stack) {
                        Some(input) => return WalkStep::Descend(input),
                        None => frame.state = phi.resolve(&mut frame.call_stack, explored),
                    }
                }
                WalkState::Settled(result) => return WalkStep::Return(result.clone()),
            }
        }
    }

    /// One step up from `value`.
    fn step(
        &mut self,
        value: MemoryValue,
        visited: &mut VisitedSet,
        call_stack: &mut CallStack,
        explored: &mut ExploredMap,
    ) -> WalkState {
        let Some(node) = self.memory_ssa.node_for_value(value) else {
            return WalkState::Settled(TraversalResult::Value(value, *call_stack));
        };
        visited.insert(value, *call_stack);
        if let Some(next) = explored.get(&value, call_stack) {
            return match next {
                ExploredResult::Advance(value, stack) => {
                    *call_stack = *stack;
                    WalkState::Walking(*value)
                }
                ExploredResult::Clobber(value) => {
                    WalkState::Settled(TraversalResult::Value(*value, *call_stack))
                }
                ExploredResult::Cycle => WalkState::Settled(TraversalResult::Cycle),
            };
        }
        match node {
            MemorySSANode::Def(def) => {
                let effects = def.effects(self.ctx);
                let mod_ref =
                    self.memory_ssa
                        .analysis_stack
                        .mod_ref(self.ctx, self.effects, &effects);
                if mod_ref.contains_mod() {
                    self.settle_clobber(value, call_stack, explored)
                } else {
                    explored.insert(
                        value,
                        *call_stack,
                        ExploredResult::Advance(def.input, *call_stack),
                    );
                    WalkState::Walking(def.input)
                }
            }
            MemorySSANode::Use(r#use) => {
                explored.insert(
                    value,
                    *call_stack,
                    ExploredResult::Advance(r#use.input, *call_stack),
                );
                WalkState::Walking(r#use.input)
            }
            MemorySSANode::CallPhi(_) | MemorySSANode::CalleePhi(_)
                if self.stop_at_phi || !self.is_interprocedural =>
            {
                self.settle_clobber(value, call_stack, explored)
            }
            MemorySSANode::Phi(_) | MemorySSANode::RegionPhi(_) if self.stop_at_phi => {
                self.settle_clobber(value, call_stack, explored)
            }
            MemorySSANode::CallPhi(_) => WalkState::AtCall(PhiWalk::new(value)),
            MemorySSANode::CalleePhi(_) if call_stack.get(self.ctx).is_empty() => {
                WalkState::Settled(TraversalResult::Value(value, *call_stack))
            }
            MemorySSANode::CalleePhi(phi) => {
                let call = call_stack.update(self.ctx, |it| it.pop().unwrap());
                let Some(taken) = phi.inputs.get(&call) else {
                    panic!("Call stack is corrupted, current caller is not a predecessor of callee")
                };
                WalkState::Walking(*taken)
            }
            MemorySSANode::Phi(_) | MemorySSANode::RegionPhi(_) => {
                WalkState::AtPhi(PhiWalk::new(value))
            }
        }
    }

    fn settle_clobber(
        &mut self,
        value: MemoryValue,
        call_stack: &mut CallStack,
        explored: &mut ExploredMap,
    ) -> WalkState {
        // Clobber in nested function is too complex to deal with downstream, so unwind the stack
        // and use the call itself as a clobber.
        if let Some(&root_call) = call_stack.get(self.ctx).first() {
            let root_value = self.memory_ssa.op_memory_def(root_call).unwrap();
            let clobber = ExploredResult::Clobber(root_value);
            explored.insert(value, *call_stack, clobber.clone());
            while let Some(call) = call_stack.update(self.ctx, |it| it.pop()) {
                let call_value = self.memory_ssa.op_memory_def(call).unwrap();
                explored.insert(call_value, *call_stack, clobber.clone());
            }
            return WalkState::Settled(TraversalResult::Value(
                root_value,
                CallStack::new(self.ctx, Default::default()),
            ));
        }

        explored.insert(value, *call_stack, ExploredResult::Clobber(value));
        WalkState::Settled(TraversalResult::Value(value, *call_stack))
    }
}

pub fn memory_ssa(
    ctx: &Context,
    root: Ptr<Operation>,
    dom_info: &mut DomInfo,
) -> Result<MemorySSA> {
    let symbol_table = &mut SymbolTableCollection::default();
    let graph_ctx = &mut MemorySSAContext::default();

    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DummySparseConstantPropagation);
    solver.load(DeadCodeAnalysis::default());
    solver.initialize_and_run(ctx, root)?;

    let mut graph_builder = MemorySSAGraphBuilder::new(graph_ctx, &solver);

    visit_all_ops_with_interface::<dyn CallableOpInterface, _>(
        ctx,
        &mut (symbol_table, dom_info, &solver, &mut graph_builder),
        root,
        |ctx, (symbol_table, dom_info, solver, graph_builder), callable| {
            let Some(callable_region) = callable.callable_region(ctx) else {
                return;
            };
            let callable_op = callable.get_operation();
            let info = MemoryOpAnalyzer::new(callable_op, dom_info, solver, symbol_table)
                .compute_info(ctx);
            graph_builder.analyze_callable(ctx, callable_region, dom_info, info);
        },
    );

    graph_builder.link_call_predecessors(ctx);

    Ok(MemorySSA {
        root,
        nodes: graph_builder.nodes,
        analysis_stack: Default::default(),
    })
}

#[pass_name]
impl Analysis for MemorySSA {
    fn compute(op: Ptr<Operation>, ctx: &Context, analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let mut dom_info = analyses.get_analysis_mut::<DomInfo>(op, ctx)?;
        memory_ssa(ctx, op, &mut dom_info)
    }
}
