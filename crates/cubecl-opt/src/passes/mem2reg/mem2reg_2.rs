use cubecl_ir::verify_op_succ;
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    derive::op_interface,
    opts::mem2reg::AllocInfo,
    region::Region,
    utils::table::IMap,
    value::Value,
};

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
        ctx: &Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut IMap<Ptr<Region>, Value>,
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
        reaching_at_block_end: &IMap<Ptr<BasicBlock>, Value>,
    ) -> Value;
}
