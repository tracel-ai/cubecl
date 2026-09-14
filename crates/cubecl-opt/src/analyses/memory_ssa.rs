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
    small_set,
};
use derive_more::From;
use derive_new::new;
use itertools::Itertools;
use pliron::{
    basic_block::BasicBlock,
    graph::{
        ControlFlowGraph, HasLabel,
        dominance::{DomFrontierMap, DomInfo},
        walkers::uninterruptible::immutable::walk_region,
    },
    linked_list::ContainsLinkedList,
    operation::OpDbg,
    printable::{self, Printable, fmt_indented_newline},
    region::Region,
    utils::table::*,
    value::DefiningEntity,
};

type MemoryEffectsMap = HMap<Ptr<Operation>, SmallSet<MemoryEffect, 4>>;
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
                    let op_effects = region_effects.entry(op).or_default();
                    op_effects.extend(effects);
                } else {
                    let region_effects = state.user_to_effects.entry(region).or_default();
                    let op_effects = region_effects.entry(op).or_default();
                    op_effects.insert(MemoryEffect::Opaque);
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
        if region.is_empty(ctx) {
            return;
        }

        let dom_tree = self.dom_info.get_dom_tree(ctx, region);
        let frontiers = DomFrontierMap::new(ctx, &region, dom_tree);

        for block in defining_blocks.iter() {
            merge_points.extend(frontiers.frontier(block));
        }
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

#[derive(new)]
pub struct MemoryDef {
    pub input: MemoryValue,
    pub effects: SmallSet<MemoryEffect, 4>,
    pub result: MemoryValue,
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

#[derive(new)]
pub struct MemoryUse {
    pub input: MemoryValue,
    pub effects: SmallSet<MemoryEffect, 4>,
    #[new(default)]
    pub is_optimized: bool,
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

fn print_effects(ctx: &Context, effects: &SmallSet<MemoryEffect, 4>) -> String {
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
                if effects.iter().any(effect_is_def) {
                    let new_res = self.ctx.new_value_at_op(op);
                    let node = MemoryDef::new(reaching_def, effects, new_res);
                    self.nodes.insert(DefiningEntity::Op(op), node.into());
                    reaching_def = new_res;
                } else {
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
        let clobber = MemoryDef::new(input, small_set![MemoryEffect::Opaque], new_reaching);
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
    fmt_indented_newline(state, f)?;
    Ok(())
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
}

pub fn memory_ssa(ctx: &Context, root: Ptr<Operation>, dom_info: &mut DomInfo) -> MemorySSA {
    let info = MemoryOpAnalyzer::new(root, dom_info).compute_info(ctx);
    let graph_ctx = &mut MemorySSAContext::default();
    let mut graph_builder = MemorySSAGraphBuilder::new(graph_ctx, root, dom_info, info);
    graph_builder.analyze_root(ctx);

    MemorySSA {
        root,
        nodes: graph_builder.nodes,
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
