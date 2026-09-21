use core::fmt;

use alloc::{
    collections::VecDeque,
    format,
    string::{String, ToString},
};
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    dialect::RegionPtrExt,
    interfaces::{
        MemoryEffect, MemoryEffects,
        memory_slot::{
            MemorySSAContext, MemorySSARegionOpInterface, MemoryValue, RegionMemoryPhiInputs,
            RegionMemoryValue,
        },
    },
    prelude::*,
};
use derive_more::From;
use derive_new::new;
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
    utils::table::*,
    value::DefiningEntity,
};

use crate::analyses::{
    alias_analysis::{AliasAnalysis, AliasAnalysisStack},
    dominance::DomFrontierCalculator,
};

type MemoryEffectsMap = HMap<Ptr<Operation>, NodeMemoryEffects>;
type RegionMemoryEffectsMap = IMap<Ptr<Region>, MemoryEffectsMap>;
type DefiningBlocks = IMap<Ptr<Region>, SmallSet<Ptr<BasicBlock>, 16>>;

type RegionSet = SmallSet<Ptr<Region>, 32>;

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
        }

        walk_region(
            ctx,
            &mut State::new(
                user_to_effects,
                defining_blocks,
                &mut regions_with_direct_def,
                &mut regions_with_direct_use,
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
                } else if let Some(mem_effects) = op_cast::<dyn MemoryEffects>(&*op.dyn_op(ctx)) {
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

#[derive(Clone, Copy)]
pub enum NodeMemoryEffects {
    Opaque,
    Op(TraitOpPtr<dyn MemoryEffects>),
}

impl NodeMemoryEffects {
    fn from_op(ctx: &Context, op: Ptr<Operation>) -> NodeMemoryEffects {
        NodeMemoryEffects::Op(TraitOpPtr::try_from_op(op, ctx).unwrap())
    }

    pub fn effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        match self {
            NodeMemoryEffects::Opaque => vec![MemoryEffect::Opaque],
            NodeMemoryEffects::Op(trait_op_ptr) => trait_op_ptr.deref(ctx).memory_effects(ctx),
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

#[derive(From)]
pub enum MemorySSANode {
    Def(MemoryDef),
    Use(MemoryUse),
    Phi(MemoryPhi),
    RegionPhi(MemoryRegionPhi),
}

impl MemorySSANode {
    pub fn input(&self) -> Option<MemoryValue> {
        match self {
            MemorySSANode::Def(def) => Some(def.input),
            MemorySSANode::Use(r#use) => Some(r#use.input),
            MemorySSANode::Phi(_) | MemorySSANode::RegionPhi(_) => None,
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
        }
    }
}

fn print_effects(ctx: &Context, effects: &NodeMemoryEffects) -> String {
    let effects = effects.effects(ctx);
    effects.iter().map(|it| it.disp(ctx).to_string()).join(", ")
}

struct MemorySSAGraphBuilder<'a> {
    root: Ptr<Operation>,
    live_on_entry: MemoryValue,

    ctx: &'a mut MemorySSAContext,
    reaching_defs: HMap<Ptr<Operation>, MemoryValue>,
    reaching_at_block_end: HMap<Ptr<BasicBlock>, MemoryValue>,
    reaching_at_region_entry: HMap<Ptr<Region>, MemoryValue>,

    nodes: HashMap<DefiningEntity, MemorySSANode>,
    pending_phis: ISet<(Ptr<BasicBlock>, MemoryValue)>,

    dom_info: &'a mut DomInfo,
    info: MemorySSAInfo,

    visited_blocks: HSet<Ptr<BasicBlock>>,
}

impl<'a> MemorySSAGraphBuilder<'a> {
    fn new(
        ctx: &'a mut MemorySSAContext,
        root: Ptr<Operation>,
        dom_info: &'a mut DomInfo,
        info: MemorySSAInfo,
    ) -> Self {
        let live_on_entry = MemoryValue::LIVE_ON_ENTRY;
        Self {
            root,
            live_on_entry,
            reaching_defs: Default::default(),
            reaching_at_block_end: Default::default(),
            reaching_at_region_entry: Default::default(),
            nodes: Default::default(),
            pending_phis: Default::default(),
            dom_info,
            info,
            visited_blocks: Default::default(),
            ctx,
        }
    }

    fn analyze_block(
        &mut self,
        ctx: &Context,
        block: Ptr<BasicBlock>,
        mut reaching_def: MemoryValue,
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
                            self.analyze_region(ctx, region, reaching_def);
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
                            self.analyze_region(ctx, region, reaching_def);
                        }

                        reaching_def = self.opaque_at(reaching_def, DefiningEntity::Op(op));
                    }
                }
            }
        }

        self.reaching_at_block_end.insert(block, reaching_def);

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

    fn analyze_region(&mut self, ctx: &Context, region: Ptr<Region>, reaching_def: MemoryValue) {
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

            reaching_def = self.analyze_block(ctx, block, reaching_def);
            let dom_tree = self.dom_info.get_dom_tree(ctx, region);

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
                let reaching_def = self.reaching_at_block_end.get(&user_block).copied();
                let reaching_def = reaching_def.unwrap_or(self.live_on_entry);

                inputs.insert(user_block, reaching_def);
            }

            let node = MemoryPhi::new(inputs, result);
            self.nodes
                .insert(DefiningEntity::Block(merge_point), node.into());
        }
    }

    fn analyze_root(&mut self, ctx: &Context) {
        let root_region = self.root.deref(ctx).get_region(0);
        // Build MemorySSA recursively through nested regions. The initial reaching
        // memory definition is the live-on-entry value.
        //
        // Nested region operations are finalized as their surrounding operations are
        // processed, after the region bodies have established their reaching memory
        // definitions.
        self.analyze_region(ctx, root_region, self.live_on_entry);

        // Finally, connect merge points to their predecessor's reaching definitions.
        self.link_merge_points(ctx);
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
        state.push_indent();
        for region in self.root.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                print_block(ctx, state, &self.nodes, block, f)?;
            }
        }
        state.pop_indent();
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
    state.push_indent();

    for op in block.deref(ctx).iter(ctx) {
        print_op(ctx, state, nodes, op, f)?;
    }

    state.pop_indent();
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
        state.push_indent();
        for region in op.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                print_block(ctx, state, nodes, block, f)?;
            }
        }
        state.pop_indent();
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
            let walker = self.walker(ctx, r#use.input, &effects);
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
        let walker = self.walker(ctx, r#use.input, &effects);
        let clobbering_access = walker.next_clobbering_def();
        let mut new_use = MemoryUse::new(clobbering_access, r#use.effects);
        new_use.is_optimized = true;
        self.nodes.insert(DefiningEntity::Op(op), new_use.into());
        Some(clobbering_access)
    }
}

pub struct MemorySSAWalker<'a> {
    stop_at_phi: bool,
    effects: &'a [MemoryEffect],
    start_value: MemoryValue,
    ctx: &'a Context,
    memory_ssa: &'a MemorySSA,
}

type VisitedSet = SmallSet<MemoryValue, 8>;

#[derive(Debug, Clone, Copy)]
enum TraversalResult {
    Value(MemoryValue),
    Cycle,
}

enum ExploredResult {
    Advance(MemoryValue),
    Clobber,
    Cycle,
}

/// One walk on the explicit stack of [`MemorySSAWalker::next_clobbering_def_impl`].
struct WalkFrame {
    state: WalkState,
    visited: VisitedSet,
}

/// Where a walk stands.
enum WalkState {
    /// Walking up from this value.
    Walking(MemoryValue),
    /// Stopped at a phi, waiting on its inputs' sub-walks.
    AtPhi(PhiWalk),
    /// Done, with this result.
    Settled(TraversalResult),
}

impl WalkFrame {
    fn new(start: MemoryValue, visited: VisitedSet) -> Self {
        Self {
            state: WalkState::Walking(start),
            visited,
        }
    }

    /// Takes the result of the sub-walk from the phi's latest input. A cycle is dropped, since
    /// cycles only happen on paths without clobbers; two inputs reaching different clobbers make
    /// the phi itself the clobber.
    fn take_input_result(
        &mut self,
        result: TraversalResult,
        explored: &mut HashMap<MemoryValue, ExploredResult>,
    ) {
        let WalkState::AtPhi(phi) = &mut self.state else {
            unreachable!("a sub-walk only runs for a phi");
        };
        let TraversalResult::Value(value) = result else {
            return;
        };
        match phi.agreed {
            None => phi.agreed = Some(value),
            Some(agreed) if agreed == value => {}
            Some(_) => {
                explored.insert(phi.value, ExploredResult::Clobber);
                self.state = WalkState::Settled(TraversalResult::Value(phi.value));
            }
        }
    }
}

/// A phi being resolved: which of its inputs has been walked, and the clobber the walked ones
/// agree on.
struct PhiWalk {
    value: MemoryValue,
    next: usize,
    agreed: Option<MemoryValue>,
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
    ) -> Option<MemoryValue> {
        while let Some(input) = self.next_input(memory_ssa) {
            self.next += 1;
            if !visited.contains(&input) {
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
            MemorySSANode::Def(_) | MemorySSANode::Use(_) => None,
        }
    }

    /// Every input walked: the clobber they agree on, or a cycle if none reached one.
    fn resolve(&self, explored: &mut HashMap<MemoryValue, ExploredResult>) -> WalkState {
        match self.agreed {
            Some(value) => {
                explored.insert(self.value, ExploredResult::Advance(value));
                WalkState::Walking(value)
            }
            None => {
                explored.insert(self.value, ExploredResult::Cycle);
                WalkState::Settled(TraversalResult::Cycle)
            }
        }
    }
}

/// What a frame needs next from the stack driving it.
enum WalkStep {
    /// A sub-walk from this phi input.
    Descend(MemoryValue),
    /// Nothing: the frame is done.
    Return(TraversalResult),
}

impl<'a> MemorySSAWalker<'a> {
    pub fn stop_at_phi(&mut self) {
        self.stop_at_phi = true;
    }

    pub fn next_clobbering_def(&self) -> MemoryValue {
        match self.next_clobbering_def_impl(&mut HashMap::new(), SmallSet::new()) {
            TraversalResult::Value(value) => value,
            TraversalResult::Cycle => self.start_value,
        }
    }

    /// Walks up from `start_value` to the access clobbering `effects`.
    ///
    /// A phi forks the walk into one sub-walk per input. The sub-walks run on an explicit stack
    /// rather than the call stack: a kernel with many rolled loops chains phis deep enough to
    /// overflow the thread compiling it.
    fn next_clobbering_def_impl(
        &self,
        explored: &mut HashMap<MemoryValue, ExploredResult>,
        visited: VisitedSet,
    ) -> TraversalResult {
        let mut stack = vec![WalkFrame::new(self.start_value, visited)];
        loop {
            let frame = stack.last_mut().expect("the root frame is popped last");
            match self.advance(frame, explored) {
                WalkStep::Descend(value) => {
                    let visited = frame.visited.clone();
                    stack.push(WalkFrame::new(value, visited));
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
    fn advance(
        &self,
        frame: &mut WalkFrame,
        explored: &mut HashMap<MemoryValue, ExploredResult>,
    ) -> WalkStep {
        loop {
            match &mut frame.state {
                WalkState::Walking(value) => {
                    frame.state = self.step(*value, &mut frame.visited, explored);
                }
                WalkState::AtPhi(phi) => {
                    match phi.next_unvisited(self.memory_ssa, &frame.visited) {
                        Some(input) => return WalkStep::Descend(input),
                        None => frame.state = phi.resolve(explored),
                    }
                }
                WalkState::Settled(result) => return WalkStep::Return(*result),
            }
        }
    }

    /// One step up from `value`.
    fn step(
        &self,
        value: MemoryValue,
        visited: &mut VisitedSet,
        explored: &mut HashMap<MemoryValue, ExploredResult>,
    ) -> WalkState {
        let Some(node) = self.memory_ssa.node_for_value(value) else {
            return WalkState::Settled(TraversalResult::Value(value));
        };
        visited.insert(value);
        if let Some(next) = explored.get(&value) {
            return match next {
                ExploredResult::Advance(next) => WalkState::Walking(*next),
                ExploredResult::Clobber => WalkState::Settled(TraversalResult::Value(value)),
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
                    explored.insert(value, ExploredResult::Clobber);
                    WalkState::Settled(TraversalResult::Value(value))
                } else {
                    explored.insert(value, ExploredResult::Advance(def.input));
                    WalkState::Walking(def.input)
                }
            }
            MemorySSANode::Use(r#use) => {
                explored.insert(value, ExploredResult::Advance(r#use.input));
                WalkState::Walking(r#use.input)
            }
            MemorySSANode::Phi(_) | MemorySSANode::RegionPhi(_) if self.stop_at_phi => {
                explored.insert(value, ExploredResult::Clobber);
                WalkState::Settled(TraversalResult::Value(value))
            }
            MemorySSANode::Phi(_) | MemorySSANode::RegionPhi(_) => {
                WalkState::AtPhi(PhiWalk::new(value))
            }
        }
    }
}

pub fn memory_ssa(ctx: &Context, root: Ptr<Operation>, dom_info: &mut DomInfo) -> MemorySSA {
    let info = MemoryOpAnalyzer::new(root, dom_info).compute_info(ctx);
    let graph_ctx = &mut MemorySSAContext::default();
    let mut graph_builder = MemorySSAGraphBuilder::new(graph_ctx, root, dom_info, info);
    graph_builder.analyze_root(ctx);

    MemorySSA {
        root,
        nodes: graph_builder.nodes,
        analysis_stack: Default::default(),
    }
}

#[pass_name]
impl Analysis for MemorySSA {
    fn compute(op: Ptr<Operation>, ctx: &Context, analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let mut dom_info = analyses.get_analysis_mut::<DomInfo>(op, ctx)?;
        Ok(memory_ssa(ctx, op, &mut dom_info))
    }
}
