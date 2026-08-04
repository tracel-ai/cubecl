//! Memory to register promotion optimization pass.

use core::{
    ops::{Deref, DerefMut},
    panic,
};

use alloc::{collections::VecDeque, rc::Rc, string::String, vec::Vec};

use cubecl_core::ir::prelude::*;
use derive_new::new;
use pliron::{
    basic_block::BasicBlock,
    common_traits::Named,
    debug_info::set_block_arg_name,
    graph::{
        ControlFlowGraph, HasLabel,
        dominance::{DomFrontierMap, DomInfo, DomTree, compute_dominator_tree},
        walkers::uninterruptible::immutable::walk_op,
    },
    irbuild::{inserter::IRInserter, listener::RecorderEvent},
    linked_list::ContainsLinkedList,
    operation::OpDbg,
    opts::mem2reg::{
        AllocInfo, PromotableAllocationInterface, PromotableOpInterface, PromotableOpKind,
    },
    region::Region,
    std_deps::hash::{FxHashMap, FxHashSet, hash_map::Entry},
};

use crate::passes::mem2reg::RegionGraphNode::{AfterOp, BeforeOp};

pub mod mem2reg_2;
pub mod scf;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RegionGraphNode {
    BeforeOp,
    Region(Ptr<Region>),
    AfterOp,
}

#[derive(Default, Debug)]
pub struct RegionGraph {
    predecessors: FxHashMap<RegionGraphNode, Vec<RegionGraphNode>>,
    successors: FxHashMap<RegionGraphNode, Vec<RegionGraphNode>>,
}

impl RegionGraph {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_successor(&mut self, pred: RegionGraphNode, succ: RegionGraphNode) {
        self.successors.entry(pred).or_default().push(succ);
        self.predecessors.entry(succ).or_default().push(pred);
    }

    pub fn get_successors(&self, pred: RegionGraphNode) -> Vec<RegionGraphNode> {
        self.successors.get(&pred).cloned().unwrap_or_default()
    }

    pub fn get_predeccessors(&self, succ: RegionGraphNode) -> Vec<RegionGraphNode> {
        self.predecessors.get(&succ).cloned().unwrap_or_default()
    }
}

impl HasLabel<Context> for RegionGraphNode {
    fn label(&self, _ctx: &Context) -> String {
        alloc::format!("{self:?}")
    }
}

impl ControlFlowGraph<Context> for RegionGraph {
    type Node = RegionGraphNode;

    fn num_successors(&self, _ctx: &Context, node: &Self::Node) -> usize {
        let Some(succs) = self.successors.get(node) else {
            return 0;
        };
        succs.len()
    }

    fn get_successor(&self, _ctx: &Context, node: &Self::Node, i: usize) -> Self::Node {
        self.successors[node][i]
    }

    fn num_predecessors(&self, _ctx: &Context, node: &Self::Node) -> usize {
        let Some(pred) = self.predecessors.get(node) else {
            return 0;
        };
        pred.len()
    }

    fn get_predecessor(&self, _ctx: &Context, node: &Self::Node, i: usize) -> Self::Node {
        self.predecessors[node][i]
    }

    fn entry_node(&self, _ctx: &Context) -> Option<Self::Node> {
        Some(RegionGraphNode::BeforeOp)
    }

    fn nodes<'a>(
        &'a self,
        _ctx: &'a Context,
    ) -> alloc::boxed::Box<dyn Iterator<Item = Self::Node> + 'a> {
        let nodes: FxHashSet<_> = self
            .successors
            .keys()
            .chain(self.predecessors.keys())
            .copied()
            .collect();
        alloc::boxed::Box::new(nodes.into_iter())
    }
}

#[derive(new, Debug)]
pub struct RegionEdge {
    pub pred: RegionGraphNode,
    pub succ: RegionGraphNode,
}

#[op_interface]
pub trait RegionDefOpInterface {
    verify_op_succ!();
    fn region_graph(&self, ctx: &Context) -> RegionGraph;

    fn get_node_argument(&self, ctx: &Context, node: RegionGraphNode, arg_idx: usize) -> Value;

    fn can_add_node_argument(&self, ctx: &Context, node: RegionGraphNode) -> bool;

    /// TODO: Docs
    fn add_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, ty: TypeHandle) -> usize;

    // TODO: Docs
    fn remove_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, arg_idx: usize);

    /// Add a new operand to be forwarded along the given edge.
    /// The operand is appended after existing operands for the specified edge.
    /// Returns the index of the newly added operand among the operands forwarded on the edge.
    /// The returned index can be used to determine the corresponding target node argument index.
    /// Panics if `edge` is invalid.
    fn add_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, operand: Value);

    /// Remove and return the operand at `opd_idx` among the operands forwarded along `edge`.
    /// Panics if `edge` or `opd_idx` is invalid.
    fn remove_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, opd_idx: usize);
}

/// A single promotable allocation: one [`AllocInfo`] from a [`PromotableAllocationInterface`] op.
#[derive(Clone, Debug)]
struct AllocCandidate {
    alloc_op: Ptr<Operation>,
    alloc_info: AllocInfo,
}

/// Collect all individual [`AllocInfo`]s from operations implementing
/// [`PromotableAllocationInterface`] rooted at `root`.
fn collect_alloc_candidates(root: Ptr<Operation>, ctx: &Context) -> Vec<AllocCandidate> {
    let mut candidates: Vec<AllocCandidate> = Vec::new();
    walk_op(
        ctx,
        &mut candidates,
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, candidates, node| {
            if let IRNode::Operation(op) = node {
                let op_obj = Operation::get_op_dyn(op, ctx);
                if let Some(iface) = op_cast::<dyn PromotableAllocationInterface>(op_obj.as_ref()) {
                    for alloc_info in iface.alloc_info(ctx) {
                        candidates.push(AllocCandidate {
                            alloc_op: op,
                            alloc_info,
                        });
                    }
                }
            }
        },
    );
    candidates
}

/// Prune allocation candidates that cannot be promoted. A candidate is removed if:
/// - any use of its pointer doesn't implement [`PromotableOpInterface`],
/// - any use reports [`PromotableOpKind::NonPromotableUse`], or
/// - any use is in a different region from the allocation.
fn prune_candidates(candidates: &mut Vec<AllocCandidate>, ctx: &Context) {
    candidates.retain(|cand| {
        cand.alloc_info.ptr.uses(ctx).iter().all(|r#use| {
            let user_op = r#use.user_op();
            let user_op_obj = Operation::get_op_dyn(user_op, ctx);
            op_cast::<dyn PromotableOpInterface>(user_op_obj.as_ref()).is_some_and(|piface| {
                let promotion_kind = piface.promotion_kind(ctx, &cand.alloc_info);

                !matches!(promotion_kind, PromotableOpKind::NonPromotableUse)
                    && has_region_def_path(ctx, cand.alloc_op, user_op)
            })
        })
    });
}

fn has_region_def_path(ctx: &Context, alloc: Ptr<Operation>, user: Ptr<Operation>) -> bool {
    let alloc_region = alloc
        .deref(ctx)
        .get_parent_region(ctx)
        .expect("Should be in region");
    let user_region = user
        .deref(ctx)
        .get_parent_region(ctx)
        .expect("Should be in region");

    let alloc_parents = region_parents(ctx, alloc_region);
    let user_parents = region_parents(ctx, user_region);

    let ncd = *user_parents
        .iter()
        .find(|reg| alloc_parents.contains(reg))
        .expect("Should have common ancestor");

    let has_path = |regions: Vec<Ptr<Region>>| {
        regions.iter().take_while(|reg| **reg != ncd).all(|reg| {
            let def = reg.deref(ctx).get_parent_op();
            def.impls::<dyn RegionDefOpInterface>(ctx)
        })
    };

    has_path(alloc_parents) && has_path(user_parents)
}

fn region_parents(ctx: &Context, mut region: Ptr<Region>) -> Vec<Ptr<Region>> {
    let mut out = vec![region];
    while let Some(parent) = region.deref(ctx).get_parent_region(ctx) {
        out.push(parent);
        region = parent;
    }
    out
}

type LiveIn = FxHashMap<Ptr<Region>, FxHashSet<Ptr<BasicBlock>>>;
type DefiningBlocks = FxHashMap<Ptr<Region>, FxHashSet<Ptr<BasicBlock>>>;
type DefiningRegions = FxHashSet<Ptr<Region>>;
type PhiRegions = FxHashMap<Ptr<Operation>, FxHashSet<RegionGraphNode>>;
type PhiBlocks = FxHashSet<Ptr<BasicBlock>>;

#[derive(Debug)]
struct Liveness {
    live_in: LiveIn,
    defining_blocks: DefiningBlocks,
    defining_regions: DefiningRegions,
    region_ops: FxHashSet<Ptr<Operation>>,
}

/// Compute per-candidate liveness anchors for mem2reg analysis.
///
/// Returns:
/// - `live_in`: blocks where the candidate's value is live-in.
/// - `defining_blocks`: blocks that contain a store defining the candidate.
fn compute_candidate_live_in_and_defining_blocks(ctx: &Context, cand: &AllocCandidate) -> Liveness {
    let ptr = cand.alloc_info.ptr;

    let mut defining_blocks = DefiningBlocks::default();
    let mut defining_regions = DefiningRegions::default();
    let mut defining_ops = FxHashSet::default();
    let mut defining_region_ops = FxHashSet::default();
    let mut live_in: LiveIn = LiveIn::default();
    let mut live_in_worklist: Vec<Ptr<BasicBlock>> = Vec::new();
    let mut regions = FxHashSet::default();

    // Compute blocks that contain uses of this pointer.
    let mut user_blocks: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
    for u in ptr.uses(ctx) {
        if let Some(block) = u.user_op().deref(ctx).get_parent_block() {
            user_blocks.insert(block);
        }
    }

    // A block is a defining block if it has any store to this candidate.
    for block in user_blocks.iter().copied() {
        let mut has_store = false;
        for op in block.deref(ctx).iter(ctx) {
            let op_obj = Operation::get_op_dyn(op, ctx);
            let Some(op_promotable) = op_cast::<dyn PromotableOpInterface>(op_obj.as_ref()) else {
                continue;
            };
            if let PromotableOpKind::Store(_) = op_promotable.promotion_kind(ctx, &cand.alloc_info)
            {
                has_store = true;
                defining_ops.insert(op);
            }
        }

        if has_store {
            let region = block
                .deref(ctx)
                .get_parent_region()
                .expect("Should be in region");
            regions.insert(region);

            let defining_blocks = defining_blocks.entry(region).or_default();
            defining_blocks.insert(block);
            if let Some(region) = block.deref(ctx).get_parent_region() {
                defining_regions.insert(region);
            }
        }
    }

    // Propagate defs to parent region ops
    let mut regions_worklist = regions.iter().copied().collect::<VecDeque<_>>();
    while let Some(region) = regions_worklist.pop_front() {
        let region_defining_blocks = defining_blocks.entry(region).or_default();
        let region_op = region.deref(ctx).get_parent_op();
        if !region_op.impls::<dyn RegionDefOpInterface>(ctx)
            || region_defining_blocks.is_empty()
            || region_op.deref(ctx).get_parent_region(ctx).is_none()
        {
            continue;
        }
        defining_ops.insert(region_op);
        defining_region_ops.insert(region_op);

        let Some(parent_block) = region_op.deref(ctx).get_parent_block() else {
            continue;
        };
        let Some(parent_region) = parent_block.deref(ctx).get_parent_region() else {
            continue;
        };
        let parent_defining_blocks = defining_blocks.entry(parent_region).or_default();
        parent_defining_blocks.insert(parent_block);

        if defining_regions.insert(parent_region) {
            regions_worklist.push_back(parent_region);
        }
    }

    // A block seeds liveness if it has a load/eliminatable use before the first store.
    for block in user_blocks {
        let mut has_store = false;
        let mut load_before_store = false;
        for op in block.deref(ctx).iter(ctx) {
            if defining_ops.contains(&op) {
                has_store = true;
            }
            let op_obj = Operation::get_op_dyn(op, ctx);
            let Some(op_promotable) = op_cast::<dyn PromotableOpInterface>(op_obj.as_ref()) else {
                continue;
            };
            match op_promotable.promotion_kind(ctx, &cand.alloc_info) {
                PromotableOpKind::Load | PromotableOpKind::EliminatableUse if !has_store => {
                    load_before_store = true;
                }
                _ => {}
            }
        }

        if has_store {
            let region = block
                .deref(ctx)
                .get_parent_region()
                .expect("Should be in region");
            regions.insert(region);

            let defining_blocks = defining_blocks.entry(region).or_default();
            defining_blocks.insert(block);
        }
        if load_before_store {
            live_in_worklist.push(block);
        }
    }

    // Propagate liveness backward through predecessors, stopping at defining blocks.
    while let Some(live_in_block) = live_in_worklist.pop() {
        let region = live_in_block
            .deref(ctx)
            .get_parent_region()
            .expect("Should be in region");
        regions.insert(region);

        let live_in = live_in.entry(region).or_default();
        let region_defining_blocks = defining_blocks.entry(region).or_default();

        if !live_in.insert(live_in_block) {
            continue;
        }
        for pred in live_in_block.preds(ctx) {
            if !region_defining_blocks.contains(&pred) {
                live_in_worklist.push(pred);
            }
        }

        // Propagate to parent region if it's live in entry
        let Some(parent_block) = region.deref(ctx).get_parent_block(ctx) else {
            continue;
        };
        let Some(parent_region) = parent_block.deref(ctx).get_parent_region() else {
            continue;
        };

        let parent_defining_blocks = defining_blocks.entry(parent_region).or_default();

        if live_in_block == region.entry_node(ctx).unwrap()
            && !parent_defining_blocks.contains(&parent_block)
        {
            live_in_worklist.push(parent_block);
        }
    }

    Liveness {
        live_in,
        defining_blocks,
        defining_regions,
        region_ops: defining_region_ops,
    }
}

/// Compute phi placement blocks for one candidate using liveness-pruned IDF.
///
/// Starting from the candidate's defining blocks, this computes the iterated
/// dominance frontier and keeps only blocks where the candidate is live-in.
fn compute_candidate_phi_blocks(
    df_map: &DomFrontierMap<Ptr<Region>, Context>,
    live_in: &FxHashSet<Ptr<BasicBlock>>,
    defining_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> FxHashSet<Ptr<BasicBlock>> {
    // Compute liveness-pruned IDF for this candidate.
    let mut phi_blocks: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
    let mut worklist: Vec<Ptr<BasicBlock>> = defining_blocks.iter().cloned().collect();
    while let Some(block) = worklist.pop() {
        for &df_block in df_map.frontier(&block) {
            // Prune early: only live-in blocks can host a useful phi.
            if !live_in.contains(&df_block) {
                continue;
            }
            if !phi_blocks.insert(df_block) {
                continue;
            }
            // Continue the IDF growth from newly inserted phi blocks.
            if !defining_blocks.contains(&df_block) {
                worklist.push(df_block);
            }
        }
    }

    phi_blocks
}

fn compute_candidate_phi_regions(
    ctx: &Context,
    region_op: Ptr<Operation>,
    defining_regions: &FxHashSet<Ptr<Region>>,
) -> FxHashSet<RegionGraphNode> {
    let op_obj = region_op.dyn_op(ctx);
    let def_op = op_cast::<dyn RegionDefOpInterface>(&*op_obj).expect("Validated when collecting");
    let graph = def_op.region_graph(ctx);
    let dom_tree = compute_dominator_tree(ctx, &graph);
    let df_map = DomFrontierMap::new(ctx, &graph, &dom_tree);

    let op_regions = region_op.regions(ctx);

    // Compute IDF for this candidate.
    let mut phi_nodes = FxHashSet::default();
    let mut worklist: Vec<_> = defining_regions.iter().cloned().collect();
    while let Some(region) = worklist.pop() {
        if !op_regions.contains(&region) {
            continue;
        }

        let node = RegionGraphNode::Region(region);
        for &df_node in df_map.frontier(&node) {
            if !phi_nodes.insert(df_node) {
                continue;
            }
            // Continue the IDF growth from newly inserted phi blocks.
            if let RegionGraphNode::Region(df_region) = df_node
                && !defining_regions.contains(&df_region)
            {
                worklist.push(df_region);
            }
        }
    }
    // Always insert after op, because regions don't automatically propagate values
    phi_nodes.insert(AfterOp);

    phi_nodes
}

fn compute_candidate_phi_nodes<F>(
    ctx: &Context,
    df_maps: &mut RegionMap<DomFrontierMap<Ptr<Region>, Context>, F>,
    liveness: &mut Liveness,
) -> (PhiRegions, PhiBlocks)
where
    F: FnMut(&Context, Ptr<Region>) -> DomFrontierMap<Ptr<Region>, Context>,
{
    let mut phi_nodes = PhiRegions::default();
    let mut phi_blocks = PhiBlocks::default();

    for &region_op in liveness.region_ops.iter() {
        let op_nodes = compute_candidate_phi_regions(ctx, region_op, &liveness.defining_regions);
        phi_nodes.insert(region_op, op_nodes);
    }

    for (&region, defining_blocks) in liveness.defining_blocks.iter() {
        let live_in = liveness.live_in.entry(region).or_default();
        let df_map = df_maps.for_region(ctx, region);
        let local_phi = compute_candidate_phi_blocks(df_map, live_in, defining_blocks);
        phi_blocks.extend(local_phi);
    }

    (phi_nodes, phi_blocks)
}

/// Prune candidates whose new block args cannot be populated because a predecessor
/// terminator does not implement [`BranchOpInterface`].
///
/// This removes both:
/// - candidates from `alloc_candidates`, and
/// - corresponding entries from `phi_blocks`.
fn prune_candidates_with_unknown_branch_from_pred(
    ctx: &Context,
    alloc_candidates: &mut Vec<AllocCandidate>,
    phi_blocks: &mut FxHashMap<Value, PhiBlocks>,
) {
    // Track invalid individual candidates by their allocation pointer.
    let mut invalid_ptrs: FxHashSet<Value> = FxHashSet::default();
    for cand in alloc_candidates.iter() {
        let ptr = cand.alloc_info.ptr;
        let invalid = phi_blocks
            .get(&ptr)
            .into_iter()
            .flatten()
            .flat_map(|&phi_block| phi_block.preds(ctx).into_iter())
            // If any predecessor terminator does not implement BranchOpInterface,
            // we won't be able to fill phi operands for this candidate, so prune it.
            .any(|pred| {
                pred.deref(ctx).get_terminator(ctx).is_none_or(|term| {
                    !op_impls::<dyn BranchOpInterface>(Operation::get_op_dyn(term, ctx).as_ref())
                })
            });
        if invalid {
            invalid_ptrs.insert(ptr);
        }
    }

    alloc_candidates.retain(|c| !invalid_ptrs.contains(&c.alloc_info.ptr));
    phi_blocks.retain(|&ptr, _| !invalid_ptrs.contains(&ptr));
}

/// Prune candidates whose new node args cannot be populated because the op doesn't support that edge.
///
/// This removes both:
/// - candidates from `alloc_candidates`, and
/// - corresponding entries from `phi_regions`.
fn prune_candidates_with_unsupported_node_argument(
    ctx: &Context,
    alloc_candidates: &mut Vec<AllocCandidate>,
    phi_regions: &mut FxHashMap<Value, PhiRegions>,
) {
    // Track invalid individual candidates by their allocation pointer.
    let mut invalid_ptrs: FxHashSet<Value> = FxHashSet::default();
    for cand in alloc_candidates.iter() {
        let ptr = cand.alloc_info.ptr;
        let invalid = phi_regions
            .get(&ptr)
            .into_iter()
            .flatten()
            .any(|(op, node)| {
                let op_obj = op.dyn_op(ctx);
                let op = op_cast::<dyn RegionDefOpInterface>(&*op_obj)
                    .expect("Validated while collecting");
                node.iter()
                    .any(|node| !op.can_add_node_argument(ctx, *node))
            });
        if invalid {
            invalid_ptrs.insert(ptr);
        }
    }

    alloc_candidates.retain(|c| !invalid_ptrs.contains(&c.alloc_info.ptr));
    phi_regions.retain(|&ptr, _| !invalid_ptrs.contains(&ptr));
}

fn get_or_create_default_def(
    alloc_cand: &AllocCandidate,
    ctx: &mut Context,
    default_defs: &mut FxHashMap<Value, Value>,
) -> Result<Value> {
    match default_defs.entry(alloc_cand.alloc_info.ptr) {
        Entry::Occupied(entry) => Ok(*entry.get()),
        Entry::Vacant(entry) => {
            let alloc_op = alloc_cand.alloc_op;
            let alloc_obj = Operation::get_op_dyn(alloc_op, ctx);
            let alloc_iface = op_cast::<dyn PromotableAllocationInterface>(alloc_obj.as_ref())
                .expect("Alloc op must implement PromotableAllocationInterface");

            // The default value must be at a place that dominates the alloc
            // and all places that the promoted value may be live at. The safest
            // such point is in the entry block, before the alloc itself.
            let alloc_block = alloc_op
                .deref(ctx)
                .get_parent_block()
                .expect("Alloc op must be in a block");
            let alloc_region = alloc_block
                .deref(ctx)
                .get_parent_region()
                .expect("Alloc op must be in a region");

            let alloc_region_entry = alloc_region
                .deref(ctx)
                .get_entry_block()
                .expect("Region must have entry block");
            let mut inserter = if alloc_region_entry == alloc_block {
                IRInserter::<Recorder>::new_before_operation(alloc_op)
            } else {
                IRInserter::<Recorder>::new_before_block_terminator(alloc_region_entry, ctx)
            };

            let default_val =
                alloc_iface.default_value(ctx, &mut inserter, &alloc_cand.alloc_info)?;
            entry.insert(default_val);
            Ok(default_val)
        }
    }
}

/// Process the events in the recorder to note down erased operations.
fn note_erased_ops(recorder: &mut Recorder, erased: &mut FxHashSet<Ptr<Operation>>) {
    for event in recorder.events.drain(..) {
        match event {
            RecorderEvent::ErasedOperation(op) => {
                erased.insert(op);
            }
            RecorderEvent::ErasedBlock(_)
            | RecorderEvent::ErasedRegion(_)
            | RecorderEvent::UnlinkedBlock(_, _) => {
                panic!("mem2reg rewrite (promotion) call backs must not alter control flow");
            }
            RecorderEvent::InsertedOperation(_)
            | RecorderEvent::InsertedBlock(_)
            | RecorderEvent::ReplacedValueUses { .. }
            | RecorderEvent::ValueTypeChanged { .. }
            | RecorderEvent::UnlinkedOperation(_, _) => {
                // No action needed for these events in this context.
            }
        }
    }
}

/// For each promotable allocation pointer, stores the current reaching definition.
type ReachingDefMap = FxHashMap<Value, Option<Value>>;
type NewRegionPhis = FxHashMap<(Ptr<Operation>, RegionGraphNode), Vec<(AllocCandidate, usize)>>;

fn outgoing_reaching_defs_for_region(
    ctx: &Context,
    def_op: &dyn RegionDefOpInterface,
    node: RegionGraphNode,
    reaching_defs: &FxHashMap<Value, Vec<Value>>,
    new_phis_in_region: &NewRegionPhis,
) -> Rc<ReachingDefMap> {
    let op = def_op.get_operation();
    let mut reaching_def_map = reaching_defs.clone();
    for &(ref cand, arg_idx) in new_phis_in_region.get(&(op, node)).into_iter().flatten() {
        let new_val = def_op.get_node_argument(ctx, node, arg_idx);
        reaching_def_map
            .get_mut(&cand.alloc_info.ptr)
            .unwrap()
            .push(new_val);
    }

    Rc::new(
        reaching_def_map
            .iter()
            .map(|(ptr, stack)| (*ptr, stack.last().copied()))
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn rename_regions<F>(
    ctx: &mut Context,
    entry_region: Ptr<Region>,
    dom_trees: &mut RegionMap<DomTree<Ptr<Region>, Context>, F>,
    new_phis_in_block: &FxHashMap<Ptr<BasicBlock>, Vec<(AllocCandidate, usize)>>,
    new_phis_in_region: &NewRegionPhis,
    root_reaching_def_map: &ReachingDefMap,
    default_def_map: &mut FxHashMap<Value, Value>,
    alloc_candidates: &[AllocCandidate],
) -> Result<()>
where
    F: FnMut(&Context, Ptr<Region>) -> DomTree<Ptr<Region>, Context>,
{
    type RegionRenameWorkItem = (Ptr<Region>, Rc<ReachingDefMap>);
    type BlockRenameWorkItem = (Ptr<BasicBlock>, Rc<ReachingDefMap>);
    let mut region_worklist: Vec<RegionRenameWorkItem> = Vec::new();
    region_worklist.push((entry_region, Rc::new(root_reaching_def_map.clone())));
    let mut processed_regions = FxHashMap::default();

    while let Some((region, region_reaching_def_map)) = region_worklist.pop() {
        let Some(entry_block) = region.entry_node(ctx) else {
            continue;
        };
        if processed_regions
            .get(&region)
            .is_some_and(|prev_defs: &Rc<_>| **prev_defs == *region_reaching_def_map)
        {
            continue;
        }
        let mut block_worklist: Vec<BlockRenameWorkItem> = Vec::new();
        block_worklist.push((entry_block, region_reaching_def_map.clone()));

        while let Some((block, incoming_reaching_def_map)) = block_worklist.pop() {
            let mut reaching_def_map = incoming_reaching_def_map
                .iter()
                .map(|(&ptr, maybe_def)| {
                    (ptr, {
                        let mut stack = Vec::new();
                        if let Some(def) = *maybe_def {
                            stack.push(def);
                        }
                        stack
                    })
                })
                .collect::<FxHashMap<_, _>>();

            // Push phi args for this block.
            for &(ref cand, arg_idx) in new_phis_in_block.get(&block).into_iter().flatten() {
                let new_val = block.deref(ctx).get_argument(arg_idx);
                reaching_def_map
                    .get_mut(&cand.alloc_info.ptr)
                    .unwrap()
                    .push(new_val);
            }

            let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
            let mut erased_ops = FxHashSet::default();
            for &op in &ops {
                if erased_ops.contains(&op) {
                    continue;
                }
                let op_obj = Operation::get_op_dyn(op, ctx);

                let regions = op.regions(ctx);
                if !regions.is_empty() {
                    match op_cast::<dyn RegionDefOpInterface>(&*op_obj) {
                        Some(def_op) => {
                            let region_graph = def_op.region_graph(ctx);
                            let successors = region_graph.get_successors(BeforeOp);
                            for &succ in successors.iter() {
                                if !processed_regions.contains_key(&region) {
                                    add_edge_operands(
                                        ctx,
                                        def_op,
                                        new_phis_in_region,
                                        default_def_map,
                                        &mut reaching_def_map,
                                        BeforeOp,
                                        succ,
                                    )?;
                                }

                                if let RegionGraphNode::Region(reg) = succ {
                                    let outgoing_reaching_def_map =
                                        outgoing_reaching_defs_for_region(
                                            ctx,
                                            def_op,
                                            succ,
                                            &reaching_def_map,
                                            new_phis_in_region,
                                        );

                                    region_worklist.push((reg, outgoing_reaching_def_map))
                                }
                            }
                        }
                        None => {
                            region_worklist.extend(
                                regions
                                    .iter()
                                    .map(|reg| (*reg, Rc::new(root_reaching_def_map.clone()))),
                            );
                        }
                    }
                }

                // Push results for this op.
                for &(ref cand, arg_idx) in
                    new_phis_in_region.get(&(op, AfterOp)).into_iter().flatten()
                {
                    let def_op = op_cast::<dyn RegionDefOpInterface>(&*op_obj)
                        .expect("Phis only exist for region def ops");
                    let new_val = def_op.get_node_argument(ctx, AfterOp, arg_idx);
                    reaching_def_map
                        .get_mut(&cand.alloc_info.ptr)
                        .unwrap()
                        .push(new_val);
                }

                let Some(piface) = op_cast::<dyn PromotableOpInterface>(op_obj.as_ref()) else {
                    continue;
                };

                let mut promote_queue = Vec::new();
                for cand in alloc_candidates {
                    let ptr = cand.alloc_info.ptr;
                    match piface.promotion_kind(ctx, &cand.alloc_info) {
                        PromotableOpKind::Load | PromotableOpKind::EliminatableUse => {
                            let reaching_def_stack = reaching_def_map.get_mut(&ptr).unwrap();
                            if reaching_def_stack.is_empty() {
                                // No reaching definition: use default value
                                let default_val =
                                    get_or_create_default_def(cand, ctx, default_def_map)?;
                                reaching_def_stack.push(default_val);
                            }
                            let current_def = *reaching_def_stack.last().unwrap();
                            promote_queue.push((cand.alloc_info.clone(), current_def));
                        }
                        PromotableOpKind::Store(stored_val) => {
                            reaching_def_map.get_mut(&ptr).unwrap().push(stored_val);
                            promote_queue.push((cand.alloc_info.clone(), stored_val));
                        }
                        // Intentionally no-op: this includes the common case where `op`
                        // does not use `cand.alloc_info.ptr` (required by the interface contract).
                        PromotableOpKind::NonPromotableUse => {}
                    }
                }

                if !promote_queue.is_empty() {
                    let rewriter = &mut IRRewriter::default();
                    rewriter.set_insertion_point_before_operation(op);
                    log::trace!("Promoting op {}", OpDbg { op, ctx });
                    piface.promote(ctx, &promote_queue, rewriter)?;
                    note_erased_ops(rewriter.get_listener_mut(), &mut erased_ops);
                }
            }

            // Fill phi operands in successor branch ops.
            let succs = block.deref(ctx).succs(ctx);
            for (succ_idx, new_phis_in_succ) in
                succs.iter().enumerate().filter_map(|(succ_idx, succ)| {
                    new_phis_in_block
                        .get(succ)
                        .map(|new_phis| (succ_idx, new_phis))
                })
            {
                let term = block
                    .deref(ctx)
                    .get_terminator(ctx)
                    .expect("Block has successors but no terminator");
                let term_obj = Operation::get_op_dyn(term, ctx);
                let branch_iface = op_cast::<dyn BranchOpInterface>(term_obj.as_ref())
                    .expect("Terminator must implement BranchOpInterface for phi blocks");
                for &(ref cand, arg_idx) in new_phis_in_succ {
                    let reaching_def_stack =
                        reaching_def_map.get_mut(&cand.alloc_info.ptr).unwrap();
                    if reaching_def_stack.is_empty() {
                        // No reaching definition: use default value
                        let default_val = get_or_create_default_def(cand, ctx, default_def_map)?;
                        reaching_def_stack.push(default_val);
                    }
                    let current_def = *reaching_def_stack.last().unwrap();
                    let succ_opd_idx =
                        branch_iface.add_successor_operand(ctx, succ_idx, current_def);
                    // The operand index returned by add_successor_operand should match the phi argument index.
                    assert!(succ_opd_idx == arg_idx, "Mismatched phi argument index");
                }
            }

            // Propagate results out of the region
            let region_op = region.deref(ctx).get_parent_op();
            let region_op_obj = region_op.dyn_op(ctx);

            if succs.is_empty()
                && let Some(def_op) = op_cast::<dyn RegionDefOpInterface>(&*region_op_obj)
            {
                let succ_graph = def_op.region_graph(ctx);
                let pred = RegionGraphNode::Region(region);
                for succ in succ_graph.get_successors(pred) {
                    if !processed_regions.contains_key(&region) {
                        add_edge_operands(
                            ctx,
                            def_op,
                            new_phis_in_region,
                            default_def_map,
                            &mut reaching_def_map,
                            pred,
                            succ,
                        )?;
                    }

                    if let RegionGraphNode::Region(succ_region) = succ {
                        let outgoing_reaching_def_map = outgoing_reaching_defs_for_region(
                            ctx,
                            def_op,
                            succ,
                            &reaching_def_map,
                            new_phis_in_region,
                        );
                        region_worklist.push((succ_region, outgoing_reaching_def_map));
                    }
                }
            }

            let outgoing_reaching_def_map = reaching_def_map
                .into_iter()
                .map(|(ptr, stack)| (ptr, stack.last().copied()))
                .collect();

            let outgoing_reaching_def_map = Rc::new(outgoing_reaching_def_map);

            let mut children: Vec<Ptr<BasicBlock>> =
                dom_trees.for_region(ctx, region).children(&block).collect();
            // Reverse so pop() preserves the original child iteration order.
            children.reverse();
            for child in children {
                block_worklist.push((child, Rc::clone(&outgoing_reaching_def_map)));
            }
        }

        processed_regions.insert(region, region_reaching_def_map);
    }

    Ok(())
}

fn add_edge_operands(
    ctx: &mut Context,
    def_op: &dyn RegionDefOpInterface,
    new_phis_in_region: &NewRegionPhis,
    default_def_map: &mut FxHashMap<Value, Value>,
    reaching_def_map: &mut FxHashMap<Value, Vec<Value>>,
    pred: RegionGraphNode,
    succ: RegionGraphNode,
) -> Result<()> {
    let region_op = def_op.get_operation();
    let new_results = new_phis_in_region
        .get(&(region_op, AfterOp))
        .cloned()
        .unwrap_or_default();
    for (cand, _) in new_results.iter() {
        let reaching_def_stack = reaching_def_map.get_mut(&cand.alloc_info.ptr).unwrap();
        if reaching_def_stack.is_empty() {
            // No reaching definition: use default value
            let default_val = get_or_create_default_def(cand, ctx, default_def_map)?;
            reaching_def_stack.push(default_val);
        }
        let current_def = *reaching_def_stack.last().unwrap();
        let edge = RegionEdge { pred, succ };
        def_op.add_edge_operand(ctx, edge, current_def);
    }
    Ok(())
}

fn prune_region_phis(ctx: &mut Context, new_phis_in_region: &NewRegionPhis) {
    for (&(op, node), results) in new_phis_in_region.iter() {
        let mut res_idxs = results.iter().map(|it| it.1).collect::<Vec<_>>();
        res_idxs.sort();

        for &res_idx in res_idxs.iter().rev() {
            let dyn_op = op.dyn_op(ctx);
            let def_op = op_cast::<dyn RegionDefOpInterface>(&*dyn_op).unwrap();
            let value = def_op.get_node_argument(ctx, node, res_idx);
            if value.is_used(ctx) {
                continue;
            }
            let graph = def_op.region_graph(ctx);
            def_op.remove_node_argument(ctx, node, res_idx);
            for pred in graph.get_predeccessors(node) {
                let edge = RegionEdge::new(pred, node);
                def_op.remove_edge_operand(ctx, edge, res_idx);
            }
        }
    }
}

struct RegionMap<V, F: FnMut(&Context, Ptr<Region>) -> V> {
    inner: FxHashMap<Ptr<Region>, V>,
    init: F,
}

impl<V, F: FnMut(&Context, Ptr<Region>) -> V> Deref for RegionMap<V, F> {
    type Target = FxHashMap<Ptr<Region>, V>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<V, F: FnMut(&Context, Ptr<Region>) -> V> DerefMut for RegionMap<V, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<V, F: FnMut(&Context, Ptr<Region>) -> V> RegionMap<V, F> {
    pub fn new(init: F) -> Self {
        Self {
            inner: Default::default(),
            init,
        }
    }

    fn for_region(&mut self, ctx: &Context, region: Ptr<Region>) -> &mut V {
        if !self.contains_key(&region) {
            let new = (self.init)(ctx, region);
            self.insert(region, new);
        }
        self.inner.get_mut(&region).unwrap()
    }
}

/// Perform memory to register promotion on regions within root.
pub fn mem2reg(root: Ptr<Operation>, ctx: &mut Context) -> Result<IRStatus> {
    let root_regions = root.regions(ctx);
    assert_eq!(root_regions.len(), 1, "Should run on single-region op");
    let entry_region = root_regions[0];

    // Collect allocations that implement the promotable allocation interface.
    let mut alloc_candidates = collect_alloc_candidates(root, ctx);

    // Prune candidates that we cannot promote based on their uses.
    prune_candidates(&mut alloc_candidates, ctx);

    if alloc_candidates.is_empty() {
        return Ok(IRStatus::Unchanged);
    }

    let mut opt_status = IRStatus::Unchanged;

    // Dominator tree and dominance frontier, once per region.
    let mut dom_trees = RegionMap::new(|ctx, region| compute_dominator_tree(ctx, &region));
    let mut df_maps = RegionMap::new(|ctx, region| {
        let dom_tree = dom_trees.for_region(ctx, region);
        DomFrontierMap::new(ctx, &region, dom_tree)
    });

    // Compute liveness and phi-placement per candidate.
    let mut phi_regions: FxHashMap<Value, PhiRegions> = FxHashMap::default();
    let mut phi_blocks: FxHashMap<Value, PhiBlocks> = FxHashMap::default();
    for cand in alloc_candidates.iter() {
        let ptr = cand.alloc_info.ptr;
        let mut liveness = compute_candidate_live_in_and_defining_blocks(ctx, cand);
        let (candidate_phi_regions, candidate_phi_blocks) =
            compute_candidate_phi_nodes(ctx, &mut df_maps, &mut liveness);
        phi_regions.insert(ptr, candidate_phi_regions);
        phi_blocks.insert(ptr, candidate_phi_blocks);
    }

    // Remove candidates where phi predecessors cannot forward values.
    prune_candidates_with_unknown_branch_from_pred(ctx, &mut alloc_candidates, &mut phi_blocks);
    prune_candidates_with_unsupported_node_argument(ctx, &mut alloc_candidates, &mut phi_regions);

    if alloc_candidates.is_empty() {
        return Ok(opt_status);
    }
    opt_status |= IRStatus::Changed;

    // Add block arguments for phis, record arg indices.
    let mut new_phis_in_block: FxHashMap<Ptr<BasicBlock>, Vec<(AllocCandidate, usize)>> =
        FxHashMap::default();
    for cand in alloc_candidates.iter() {
        let ptr = cand.alloc_info.ptr;
        if let Some(needed_blocks) = phi_blocks.get(&ptr) {
            let needed_blocks: Vec<Ptr<BasicBlock>> = needed_blocks.iter().cloned().collect();
            for phi_block in needed_blocks {
                let arg_idx = BasicBlock::push_argument(phi_block, ctx, cand.alloc_info.ty);
                set_block_arg_name(ctx, phi_block, arg_idx, ptr.given_name(ctx));
                new_phis_in_block
                    .entry(phi_block)
                    .or_default()
                    .push((cand.clone(), arg_idx));
            }
        }
    }

    // Add region op results for region ops, record res indices.
    let mut new_phis_in_regions: NewRegionPhis = FxHashMap::default();
    for cand in alloc_candidates.iter() {
        let ptr = cand.alloc_info.ptr;
        if let Some(needed_nodes) = phi_regions.get(&ptr) {
            for (&region_op, needed_nodes) in needed_nodes.iter() {
                let op_obj = region_op.dyn_op(ctx);
                let def_op = op_cast::<dyn RegionDefOpInterface>(&*op_obj)
                    .expect("Validated when collecting");
                for &phi_node in needed_nodes {
                    let res_idx = def_op.add_node_argument(ctx, phi_node, cand.alloc_info.ty);
                    new_phis_in_regions
                        .entry((region_op, phi_node))
                        .or_default()
                        .push((cand.clone(), res_idx));
                }
            }
        }
    }

    // Initialize reaching def map for this region's candidates.
    let reaching_def_map: ReachingDefMap = alloc_candidates
        .iter()
        .map(|c| (c.alloc_info.ptr, None))
        .collect();
    let mut default_def_map: FxHashMap<Value, Value> = FxHashMap::default();

    // SSA rename via recursive dominator tree walk, starting from the entry region and traversing
    // into child regions.
    rename_regions(
        ctx,
        entry_region,
        &mut dom_trees,
        &new_phis_in_block,
        &new_phis_in_regions,
        &reaching_def_map,
        &mut default_def_map,
        &alloc_candidates,
    )?;

    // "Promote" (remove) the allocations themselves. Group them into
    // a single promote call per alloc op and then invoke the interface method.
    let mut alloc_op_to_infos: FxHashMap<Ptr<Operation>, Vec<AllocInfo>> = FxHashMap::default();
    let rewriter = &mut IRRewriter::default();

    for cand in alloc_candidates.iter() {
        alloc_op_to_infos
            .entry(cand.alloc_op)
            .or_default()
            .push(cand.alloc_info.clone());
    }

    let mut erased_ops = FxHashSet::default();
    for (op, infos) in alloc_op_to_infos {
        if erased_ops.contains(&op) {
            panic!("Alloc op was already erased during promotion of another candidate");
        }
        rewriter.set_insertion_point_before_operation(op);
        let op = Operation::get_op_dyn(op, ctx);
        let piface = op_cast::<dyn PromotableAllocationInterface>(op.as_ref())
            .expect("Alloc op must implement PromotableAllocationInterface");
        log::debug!(
            "Promoting allocation {}",
            OpDbg {
                op: op.get_operation(),
                ctx
            }
        );
        piface.promote(ctx, rewriter, &infos)?;
        note_erased_ops(rewriter.get_listener_mut(), &mut erased_ops);
    }

    prune_region_phis(ctx, &new_phis_in_regions);

    Ok(opt_status)
}

#[derive(Default)]
/// The mem2reg pass, which promotes memory allocations to SSA registers where possible.
pub struct Mem2RegPass;

impl Pass for Mem2RegPass {
    fn name(&self) -> &str {
        "mem2reg"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut pass_res = PassResult::default();
        // Run mem2reg on the entire operation tree rooted at `op`
        pass_res.ir_changed |= mem2reg(op, ctx)?;
        // mem2reg does not touch the CFG structure, so we can preserve dominator info if it exists.
        pass_res.set_preserved::<DomInfo>();
        Ok(pass_res)
    }
}
