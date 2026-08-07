use core::cmp::Ordering;

use alloc::{collections::VecDeque, vec::Vec};
use cubecl_ir::{interfaces::aliasing::PointerExt, prelude::*, verify_op_succ};
use derive_new::new;
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    derive::op_interface,
    graph::{
        ControlFlowGraph,
        dominance::{DomFrontierMap, DomInfo},
        walkers::uninterruptible::immutable::walk_region,
    },
    irbuild::listener::DummyListener,
    linked_list::ContainsLinkedList,
    operation::Operation,
    opts::mem2reg::{
        AllocInfo, PromotableAllocationInterface, PromotableOpInterface, PromotableOpKind,
    },
    region::Region,
    utils::{
        table::{HMap, HSet, IMap, ISet, SmallMap, SmallSet},
        vec_exns::VecExtns,
    },
    value::{Use, Value},
};

use crate::analyses::slices::get_value_forward_slice;

pub mod scf;

#[op_interface]
pub trait PromotableRegionOpInterface {
    verify_op_succ!();

    /// Returns true if `region` (a child of this op) can be analysed for
    /// promotion with respect to `alloc`.
    /// `hasValueStores` is a hint: true when the region contains stores to alloc.
    fn is_region_promotable(
        &self,
        ctx: &Context,
        alloc: &AllocInfo,
        region: Ptr<Region>,
        has_value_stores: bool,
    ) -> bool;

    /// Called before descending into nested regions.
    /// `reachingDef` is the value in `slot` on entry to this op.
    /// Populate `regionsToProcess` with the reaching def each region starts with.
    /// You may mutate the op in place, but do NOT delete ops or touch terminators.
    fn setup_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    );

    /// Called after reaching defs are computed for all regions, but before
    /// blocking uses are removed. Returns the new reaching def at the op's exit.
    /// Mutation is allowed, but you must not change control flow or add ops that
    /// interact with the slot's value.
    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value;
}

type BlockingUsesMap = IMap<Ptr<Operation>, SmallSet<Use<Value>, 4>>;
type RegionBlockingUsesMap = SmallMap<Ptr<Region>, BlockingUsesMap, 2>;
type LogicalResult = core::result::Result<(), ()>;

type RegionSet = SmallSet<Ptr<Region>, 32>;

#[derive(Default)]
struct RegionPromotionInfo {
    has_value_stores: bool,
}

#[derive(Default)]
struct AllocPromotionInfo {
    merge_points: ISet<Ptr<BasicBlock>>,
    user_to_blocking_uses: RegionBlockingUsesMap,
    regions_to_promote: IMap<Ptr<Region>, RegionPromotionInfo>,
}

#[derive(new)]
struct AllocPromotionAnalyzer<'a> {
    alloc: &'a AllocInfo,
    dom_info: &'a mut DomInfo,
}

impl AllocPromotionAnalyzer<'_> {
    fn compute_blocking_uses(
        &mut self,
        ctx: &mut Context,
        user_to_blocking_uses: &mut RegionBlockingUsesMap,
        regions_to_promote: &mut IMap<Ptr<Region>, RegionPromotionInfo>,
    ) -> LogicalResult {
        let ptr_op = self.alloc.ptr.defining_op().unwrap();
        let ptr_region = ptr_op.deref(ctx).get_parent_region(ctx).unwrap();

        for r#use in self.alloc.ptr.uses(ctx) {
            let region = r#use.user_op().deref(ctx).get_parent_region(ctx).unwrap();
            let region_blocking_uses = user_to_blocking_uses.entry(region).or_default();
            let blocking_uses = region_blocking_uses.entry(r#use.user_op()).or_default();
            blocking_uses.insert(r#use);
        }

        let mut regions_with_direct_use = RegionSet::new();
        let mut regions_with_direct_store = RegionSet::new();

        let forward_slice = get_value_forward_slice(ctx, self.alloc.ptr);
        for user in forward_slice {
            let user_region = user.deref(ctx).get_parent_region(ctx).unwrap();
            let user_obj = user.dyn_op(ctx);
            let Some(blocking_uses_map) = user_to_blocking_uses.get(&user_region) else {
                continue;
            };
            if !blocking_uses_map.contains_key(&user) {
                continue;
            }

            if let Some(promotable) = op_cast::<dyn PromotableOpInterface>(&*user_obj) {
                match promotable.promotion_kind(ctx, self.alloc) {
                    PromotableOpKind::Load => {
                        regions_with_direct_use.insert(user_region);
                    }
                    PromotableOpKind::Store(_) => {
                        regions_with_direct_store.insert(user_region);
                    }
                    PromotableOpKind::EliminatableUse => {}
                    PromotableOpKind::NonPromotableUse => return Err(()),
                }
            } else {
                // An operation that has blocking uses must be promoted. If it is not
                // promotable, promotion must fail.
                return Err(());
            }
        }

        let mut visit_regions = |regions_to_propagate_from: &mut VecDeque<Ptr<Region>>,
                                 has_value_stores: bool|
         -> LogicalResult {
            while let Some(region) = regions_to_propagate_from.pop_back() {
                if region == ptr_region || regions_to_promote.contains_key(&region) {
                    continue;
                }

                regions_to_promote.insert(region, RegionPromotionInfo { has_value_stores });

                let parent_op = region.deref(ctx).get_parent_op().dyn_op(ctx);
                let Some(promotable_parent_op) =
                    op_cast::<dyn PromotableRegionOpInterface>(&*parent_op)
                else {
                    return Err(());
                };

                if !promotable_parent_op.is_region_promotable(
                    ctx,
                    self.alloc,
                    region,
                    has_value_stores,
                ) {
                    return Err(());
                }

                regions_to_propagate_from
                    .push_back(region.deref(ctx).get_parent_region(ctx).unwrap());
            }

            Ok(())
        };

        let mut regions_to_propagate_from = VecDeque::new();
        regions_to_propagate_from.extend(regions_with_direct_store);
        visit_regions(&mut regions_to_propagate_from, true)?;

        regions_to_propagate_from.clear();
        regions_to_propagate_from.extend(regions_with_direct_use);
        visit_regions(&mut regions_to_propagate_from, false)?;

        Ok(())
    }

    fn compute_merge_points(
        &mut self,
        ctx: &mut Context,
        region: Ptr<Region>,
        defining_blocks: &SmallSet<Ptr<BasicBlock>, 16>,
        merge_points: &mut ISet<Ptr<BasicBlock>>,
    ) {
        if region.deref(ctx).iter(ctx).count() == 1 {
            return;
        }

        let dom_tree = self.dom_info.get_dom_tree(ctx, region);
        let frontiers = DomFrontierMap::new(ctx, &region, dom_tree);

        for block in defining_blocks.iter() {
            merge_points.extend(frontiers.frontier(block));
        }
    }

    fn are_merge_points_usable(&self, ctx: &Context, merge_points: &ISet<Ptr<BasicBlock>>) -> bool {
        for merge_point in merge_points.iter() {
            for pred in merge_point.preds(ctx) {
                let term = pred.deref(ctx).get_terminator(ctx);
                if !term.is_some_and(|it| it.impls::<dyn BranchOpInterface>(ctx)) {
                    return false;
                }
            }
        }
        true
    }

    fn compute_info(&mut self, ctx: &mut Context) -> Option<AllocPromotionInfo> {
        let mut info = AllocPromotionInfo::default();

        self.compute_blocking_uses(
            ctx,
            &mut info.user_to_blocking_uses,
            &mut info.regions_to_promote,
        )
        .ok()?;

        let mut defining_blocks = IMap::<_, SmallSet<_, 16>>::default();
        for r#use in self.alloc.ptr.uses(ctx) {
            let user = r#use.user_op();
            let user_dyn = user.dyn_op(ctx);
            let block = user.deref(ctx).get_parent_block().unwrap();
            let region = user.deref(ctx).get_parent_region(ctx).unwrap();
            if let Some(promotable) = op_cast::<dyn PromotableOpInterface>(&*user_dyn) {
                let kind = promotable.promotion_kind(ctx, self.alloc);
                if matches!(kind, PromotableOpKind::Store(_)) {
                    defining_blocks.entry(region).or_default().insert(block);
                }
            }
        }

        for (region, region_info) in info.regions_to_promote.iter() {
            if region_info.has_value_stores {
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

        if !self.are_merge_points_usable(ctx, &info.merge_points) {
            return None;
        }

        Some(info)
    }
}

type BlockIndexCache = HMap<Ptr<BasicBlock>, HMap<Ptr<BasicBlock>, usize>>;

struct AllocPromoter<'a> {
    alloc: AllocInfo,
    allocator: &'a TraitOp<dyn PromotableAllocationInterface>,

    default_value: Option<Value>,

    reaching_defs: HMap<TraitOp<dyn PromotableOpInterface>, Option<Value>>,
    replaced_values_map: HMap<TraitOp<dyn PromotableOpInterface>, Value>,

    reaching_at_block_end: HMap<Ptr<BasicBlock>, Value>,

    rewriter: IRRewriter<DummyListener>,

    dom_info: &'a mut DomInfo,
    info: AllocPromotionInfo,

    visited_blocks: HSet<Ptr<BasicBlock>>,

    block_index_cache: &'a mut BlockIndexCache,
}

impl<'a> AllocPromoter<'a> {
    fn new(
        alloc: AllocInfo,
        allocator: &'a TraitOp<dyn PromotableAllocationInterface>,
        dom_info: &'a mut DomInfo,
        info: AllocPromotionInfo,
        block_index_cache: &'a mut BlockIndexCache,
    ) -> Self {
        Self {
            alloc,
            allocator,
            default_value: None,
            rewriter: Default::default(),
            reaching_defs: Default::default(),
            replaced_values_map: Default::default(),
            reaching_at_block_end: Default::default(),
            dom_info,
            info,
            block_index_cache,
            visited_blocks: Default::default(),
        }
    }

    fn get_or_create_default_value(&mut self, ctx: &mut Context) -> Result<Value> {
        if let Some(default_val) = self.default_value {
            return Ok(default_val);
        }
        let mut rewriter = IRRewriter::<DummyListener>::default();
        let block = self.alloc.ptr.get_defining_block(ctx);
        rewriter.set_insertion_point_to_block_start(block.expect("Should have parent"));
        let value = self
            .allocator
            .default_value(ctx, &mut rewriter, &self.alloc)?;
        self.default_value = Some(value);
        Ok(value)
    }

    fn promote_in_block(
        &mut self,
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        mut reaching_def: Option<Value>,
    ) -> Option<Value> {
        if self.visited_blocks.contains(&block) {
            panic!("promote_in_block was called twice on one block");
        }
        self.visited_blocks.insert(block);

        let block_ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in block_ops {
            if let Some(promotable) = TraitOp::<dyn PromotableOpInterface>::try_from_op(op, ctx) {
                let parent_region = op.deref(ctx).get_parent_region(ctx).unwrap();
                let region_blocking_uses = self.info.user_to_blocking_uses.entry(parent_region);
                if region_blocking_uses.or_default().contains_key(&op) {
                    self.reaching_defs.insert(promotable.clone(), reaching_def);
                }

                if let PromotableOpKind::Store(stored) = promotable.promotion_kind(ctx, &self.alloc)
                {
                    reaching_def = Some(stored);
                    self.replaced_values_map.insert(promotable, stored);
                }
            }

            let op_obj = op.dyn_op(ctx);
            if let Some(promotable_region_op) = op_cast::<dyn PromotableRegionOpInterface>(&*op_obj)
            {
                let mut needs_promotion = false;
                let mut has_value_stores = false;
                for region in op.regions(ctx) {
                    let Some(region_info) = self.info.regions_to_promote.get(&region) else {
                        continue;
                    };
                    needs_promotion = true;
                    if !region_info.has_value_stores {
                        continue;
                    }
                    has_value_stores = true;
                    break;
                }

                if needs_promotion {
                    let mut regions_to_process = SmallMap::new();

                    let reaching = *reaching_def
                        .get_or_insert_with(|| self.get_or_create_default_value(ctx).unwrap());
                    promotable_region_op.setup_promotion(
                        ctx,
                        &self.alloc,
                        reaching,
                        has_value_stores,
                        &mut regions_to_process,
                    );

                    for (&region, &reaching_def) in regions_to_process.iter() {
                        if !self.info.regions_to_promote.contains_key(&region) {
                            continue;
                        }
                        self.promote_in_region(ctx, region, Some(reaching_def));
                    }

                    self.rewriter.set_insertion_point_after_operation(op);
                    reaching_def = Some(promotable_region_op.finalize_promotion(
                        ctx,
                        &self.alloc,
                        reaching,
                        has_value_stores,
                        &self.reaching_at_block_end,
                    ));

                    for (region, _) in regions_to_process {
                        self.remove_blocking_uses(ctx, region);
                    }
                }
            }
        }

        if let Some(reaching_def) = reaching_def {
            self.reaching_at_block_end.insert(block, reaching_def);
        }
        reaching_def
    }

    fn promote_in_region(
        &mut self,
        ctx: &mut Context,
        region: Ptr<Region>,
        reaching_def: Option<Value>,
    ) {
        if region.deref(ctx).iter(ctx).count() == 1 {
            let entry = region.entry_node(ctx).unwrap();
            self.promote_in_block(ctx, entry, reaching_def);
            return;
        }

        let entry = region.entry_node(ctx).unwrap();

        let mut dfs_stack = VecDeque::new();

        dfs_stack.push_back((entry, reaching_def));

        while let Some((block, mut reaching_def)) = dfs_stack.pop_back() {
            if self.info.merge_points.contains(&block) {
                let arg_idx = BasicBlock::push_argument(block, ctx, self.alloc.ty);
                reaching_def = Some(block.deref(ctx).get_argument(arg_idx));
            }

            reaching_def = self.promote_in_block(ctx, block, reaching_def);
            let dom_tree = self.dom_info.get_dom_tree(ctx, region);

            for child in dom_tree.children(&block) {
                dfs_stack.push_back((child, reaching_def));
            }
        }
    }

    fn remove_blocking_uses(&mut self, ctx: &mut Context, mut region: Ptr<Region>) {
        let Some(blocking_uses_map) = self.info.user_to_blocking_uses.get_mut(&region) else {
            return;
        };
        let Some(first) = blocking_uses_map.keys().next().copied() else {
            return;
        };

        // Operations may have been moved to a different region at this point.
        // To cover this, we process the current region of an operation to remove
        // instead of the provided region.
        region = first.deref(ctx).get_parent_region(ctx).unwrap();

        let mut users_to_remove_uses = blocking_uses_map.keys().copied().collect::<Vec<_>>();

        dominance_sort(
            ctx,
            &mut users_to_remove_uses,
            region,
            self.dom_info,
            self.block_index_cache,
        );

        for to_promote in users_to_remove_uses.into_iter().rev() {
            if let Some(promotable) =
                TraitOp::<dyn PromotableOpInterface>::try_from_op(to_promote, ctx)
            {
                self.rewriter
                    .set_insertion_point_before_operation(to_promote);
                let reaching_def = self.reaching_defs.get(&promotable).copied().flatten();
                let reaching_def = reaching_def.unwrap_or_else(|| {
                    get_or_create_default_value(
                        &mut self.default_value,
                        self.allocator,
                        &self.alloc,
                        ctx,
                    )
                    .unwrap()
                });
                promotable
                    .promote(
                        ctx,
                        &[(self.alloc.clone(), reaching_def)],
                        &mut self.rewriter,
                    )
                    .unwrap();
                blocking_uses_map.shift_remove(&to_promote);
                continue;
            }
        }
    }

    fn link_merge_points(&mut self, ctx: &mut Context) {
        // We want to eliminate unused block arguments. In case connecting a block
        // argument to its predecessor would trigger the use of the predecessor's
        // unused block argument, we need to process merge points in an expanding
        // worklist, `merge_point_args_to_process`.

        let mut merge_point_args_unused = SmallSet::<_, 8>::new();
        let mut merge_point_args_to_process = Vec::new();
        for merge_point in self.info.merge_points.iter() {
            let arg = merge_point.deref(ctx).arguments().last().unwrap();
            if !arg.is_used(ctx) {
                merge_point_args_unused.insert(arg);
            } else {
                merge_point_args_to_process.push(arg);
            }
        }

        while let Some(arg) = merge_point_args_to_process.pop() {
            let merge_point = arg.get_defining_block(ctx).unwrap();

            for r#use in merge_point.uses(ctx) {
                let user_block = r#use.user_op().deref(ctx).get_parent_block().unwrap();
                let reaching_def = self.reaching_at_block_end.get(&user_block).copied();
                let reaching_def =
                    reaching_def.unwrap_or_else(|| self.get_or_create_default_value(ctx).unwrap());

                // If the reaching definition is a block argument of an unused merge
                // point, mark it as used and process it as such later.
                if reaching_def.defining_block().is_some()
                    && merge_point_args_unused.remove(&reaching_def)
                {
                    merge_point_args_to_process.push_back(reaching_def);
                }

                let user = TraitOp::<dyn BranchOpInterface>::try_from_op(r#use.user_op(), ctx)
                    .expect("Already filtered by analysis if not implemented");
                let succ_idx = r#use.try_find_index(ctx).unwrap();
                user.add_successor_operand(ctx, succ_idx, reaching_def);
            }

            self.rewriter
                .set_insertion_point_to_block_start(merge_point);
        }

        for arg in merge_point_args_unused {
            let merge_point = arg.get_defining_block(ctx).unwrap();
            let num_args = merge_point.deref(ctx).get_num_arguments();
            BasicBlock::remove_argument(merge_point, ctx, num_args - 1);
        }
    }

    fn promote_alloc(&mut self, ctx: &mut Context) -> Result<()> {
        let alloc_op = self.alloc.ptr.get_root_defining_op(ctx).unwrap();
        let ptr_region = alloc_op.deref(ctx).get_parent_region(ctx).unwrap();
        // Perform the promotion recursively through nested regions. The reaching
        // definition starts with a None value that will be replaced by a
        // lazily-created default value if the value must be passed to a promotion
        // interface while no store has been encountered yet.
        // Innermost regions will see their blocking uses be removed, but not the
        // outermost region which we have to remove manually afterwards. This is
        // because `PromotableRegionOpInterface::finalize_promotion` must be called
        // before remove_blocking_uses.
        self.promote_in_region(ctx, ptr_region, None);

        // Blocking uses can then be removed for the outermost region.
        self.remove_blocking_uses(ctx, ptr_region);

        // Finally, connect merge points to their predecessor's reaching definitions.
        self.link_merge_points(ctx);

        self.allocator
            .promote(ctx, &mut self.rewriter, core::slice::from_ref(&self.alloc))
    }
}

fn get_or_create_default_value(
    default_value: &mut Option<Value>,
    allocator: &TraitOp<dyn PromotableAllocationInterface>,
    alloc: &AllocInfo,
    ctx: &mut Context,
) -> Result<Value> {
    if let Some(default_val) = *default_value {
        return Ok(default_val);
    }
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let block = alloc.ptr.get_defining_block(ctx);
    rewriter.set_insertion_point_to_block_start(block.expect("Should have parent"));
    let value = allocator.default_value(ctx, &mut rewriter, alloc)?;
    *default_value = Some(value);
    Ok(value)
}

fn get_or_create_block_indices<'a>(
    ctx: &Context,
    dom_info: &mut DomInfo,
    block_index_cache: &'a mut BlockIndexCache,
    region_entry_block: Ptr<BasicBlock>,
) -> &'a HMap<Ptr<BasicBlock>, usize> {
    // There's a borrow checker bug here that I've seen discussed on the rustlang Zulip. Need to
    // work around it like this.
    // It should allow mutable borrows in mutually exclusive branches but currently doesn't.
    let exists = block_index_cache.contains_key(&region_entry_block);
    let block_indices = block_index_cache.entry(region_entry_block).or_default();
    if !exists {
        let parent_region = region_entry_block.deref(ctx).get_parent_region().unwrap();
        let topological_order = get_blocks_sorted_by_dominance(ctx, dom_info, parent_region);
        for (index, block) in topological_order.into_iter().enumerate() {
            block_indices.insert(block, index);
        }
    }
    block_indices
}

/// Sorts `ops` according to dominance. Relies on the topological order of basic
/// blocks to get a deterministic ordering. Uses `block_index_cache` to avoid the
/// potentially expensive recomputation of a block index map.
/// This function assumes no blocks are ever deleted or entry block changed
/// during the lifetime of the block index cache.
fn dominance_sort(
    ctx: &Context,
    ops: &mut [Ptr<Operation>],
    region: Ptr<Region>,
    dom_info: &mut DomInfo,
    block_index_cache: &mut BlockIndexCache,
) {
    if region.deref(ctx).iter(ctx).count() == 0 {
        return;
    }

    let entry = region.deref(ctx).get_entry_block().unwrap();
    let topo_block_indices = get_or_create_block_indices(ctx, dom_info, block_index_cache, entry);

    ops.sort_by(|lhs, rhs| {
        let lhs_block_index = topo_block_indices[&lhs.deref(ctx).get_parent_block().unwrap()];
        let rhs_block_index = topo_block_indices[&rhs.deref(ctx).get_parent_block().unwrap()];
        match lhs_block_index.cmp(&rhs_block_index) {
            ord @ (Ordering::Less | Ordering::Greater) => ord,
            Ordering::Equal => match dom_info.op_strictly_dominates_op(ctx, *lhs, *rhs) {
                true => Ordering::Less,
                false => Ordering::Greater,
            },
        }
    });
}

fn get_blocks_sorted_by_dominance(
    ctx: &Context,
    dom_info: &mut DomInfo,
    region: Ptr<Region>,
) -> ISet<Ptr<BasicBlock>> {
    let dom_tree = dom_info.get_dom_tree(ctx, region);
    let mut out = ISet::default();
    let mut worklist = vec![];
    worklist.extend(dom_tree.root());

    while let Some(block) = worklist.pop() {
        if !out.insert(block) {
            continue;
        };
        let children = dom_tree.children(&block);
        worklist.extend(children);
    }
    out
}

fn try_promote_allocs(
    ctx: &mut Context,
    allocators: Vec<TraitOp<dyn PromotableAllocationInterface>>,
    dom_info: &mut DomInfo,
) -> Result<bool> {
    let mut promoted_any = false;

    // A cache that stores deterministic block indices which are used to determine
    // a valid operation modification order. The block index maps are computed
    // lazily and cached to avoid expensive recomputation.
    let mut block_index_cache = BlockIndexCache::default();

    let mut worklist = allocators;

    let mut new_worklist = Vec::with_capacity(worklist.len());
    // Fix point loop
    loop {
        let mut changes_in_this_round = false;
        for allocator in worklist.drain(..) {
            let mut changed_allocator = false;
            for alloc in allocator.alloc_info(ctx) {
                if !alloc.ptr.is_used(ctx) {
                    continue;
                }

                let Some(info) = AllocPromotionAnalyzer::new(&alloc, dom_info).compute_info(ctx)
                else {
                    continue;
                };
                let mut promoter =
                    AllocPromoter::new(alloc, &allocator, dom_info, info, &mut block_index_cache);
                promoter.promote_alloc(ctx)?;
                changed_allocator = true;
                break;
            }
            if !changed_allocator {
                new_worklist.push(allocator);
            }
            changes_in_this_round |= changed_allocator;
        }
        if !changes_in_this_round {
            break;
        }
        promoted_any = true;

        core::mem::swap(&mut worklist, &mut new_worklist);
        new_worklist.clear();
    }

    Ok(promoted_any)
}

pub struct Mem2RegPass;

#[pass_name]
impl Pass for Mem2RegPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let mut dom_info = analyses.get_analysis_mut::<DomInfo>(op, ctx)?;
        for region in op.regions(ctx) {
            if region.entry_node(ctx).is_none() {
                continue;
            }

            let mut allocators = Vec::new();
            walk_region(
                ctx,
                &mut allocators,
                &WALKCONFIG_ANY,
                region,
                |ctx, allocators, node| {
                    if let IRNode::Operation(op) = node
                        && let Some(allocator) = TraitOp::try_from_op(op, ctx)
                    {
                        allocators.push(allocator);
                    }
                },
            );

            if try_promote_allocs(ctx, allocators, &mut dom_info)? {
                res.ir_changed |= IRStatus::Changed;
            }
        }

        Ok(res)
    }
}
