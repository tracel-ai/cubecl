use core::{
    fmt::{self, Display},
    hash::Hash,
};

use derive_more::{Eq, PartialEq};
use derive_new::new;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    graph::HasLabel,
    opts::mem2reg::AllocInfo,
    printable::{self, Printable},
    region::Region,
    utils::table::{HMap, SmallMap, SmallSet},
    value::DefiningEntity,
};

use crate::prelude::*;

pub type LogicalResult = core::result::Result<(), ()>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DeletionKind {
    Keep,
    Delete,
}

/// The defining entity of a [`MemoryValue`]:
/// Either an [`Operation`], a [`BasicBlock`] or the special live-on-entry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDefiningEntity {
    Op(Ptr<Operation>),
    Block(Ptr<BasicBlock>),
    LiveOnEntry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryValue {
    val_uid: u64,
    #[eq(skip)]
    defining_entity: MemoryDefiningEntity,
}

impl Hash for MemoryValue {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.val_uid.hash(state);
    }
}

impl Display for MemoryValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &MemoryValue::LIVE_ON_ENTRY {
            write!(f, "LiveOnEntry")
        } else {
            write!(f, "M{}", self.val_uid)
        }
    }
}

impl MemoryValue {
    pub const LIVE_ON_ENTRY: MemoryValue = MemoryValue {
        val_uid: 0,
        defining_entity: MemoryDefiningEntity::LiveOnEntry,
    };

    pub fn defining_entity(&self) -> Option<DefiningEntity> {
        match self.defining_entity {
            MemoryDefiningEntity::Op(op) => Some(DefiningEntity::Op(op)),
            MemoryDefiningEntity::Block(block) => Some(DefiningEntity::Block(block)),
            MemoryDefiningEntity::LiveOnEntry => None,
        }
    }
}

#[derive(Default)]
pub struct MemorySSAContext {
    last_idx: u64,
}

impl MemorySSAContext {
    pub fn new_value_in_block(&mut self, block: Ptr<BasicBlock>) -> MemoryValue {
        self.last_idx += 1;
        MemoryValue {
            val_uid: self.last_idx,
            defining_entity: MemoryDefiningEntity::Block(block),
        }
    }

    pub fn new_value_at_op(&mut self, op: Ptr<Operation>) -> MemoryValue {
        self.last_idx += 1;
        MemoryValue {
            val_uid: self.last_idx,
            defining_entity: MemoryDefiningEntity::Op(op),
        }
    }
}

/// Different from `RegionPredecessor` because terminators don't actually matter for memory SSA
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegionPredecessor {
    Parent,
    Block(Ptr<BasicBlock>),
}

impl Printable for MemoryRegionPredecessor {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryRegionPredecessor::Parent => f.write_str("Parent"),
            MemoryRegionPredecessor::Block(block) => {
                write!(f, "{}", block.label(ctx))
            }
        }
    }
}

pub type RegionMemoryPhiInputs = SmallMap<MemoryRegionPredecessor, MemoryValue, 2>;
pub enum RegionMemoryValue {
    Forward(MemoryValue),
    RegionPhi(RegionMemoryPhiInputs),
}

#[op_interface]
pub trait PromotableRegionOpInterface {
    verify_op_succ!();

    /// Returns true if `region` (a child of this op) can be analysed for
    /// promotion with respect to `alloc`.
    /// `has_value_stores` is a hint: true when the region contains stores to alloc.
    fn is_region_promotable(
        &self,
        ctx: &Context,
        alloc: &AllocInfo,
        region: Ptr<Region>,
        has_value_stores: bool,
    ) -> bool;

    /// Called before descending into nested regions.
    /// `reaching_def` is the value in `slot` on entry to this op.
    /// Populate `regions_to_process` with the reaching def each region starts with.
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

#[op_interface]
pub trait MemorySSARegionOpInterface {
    verify_op_succ!();

    /// Called before descending into nested regions.
    /// `reaching_def` is the state of the memory on entry to this op.
    /// Populate `regions_to_process` with the reaching def each region starts with.
    fn setup_memory_ssa(
        &self,
        ctx: &Context,
        state: &mut MemorySSAContext,
        reaching_def: MemoryValue,
        has_memory_defs: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, MemoryValue, 2>,
    );

    /// Called after reaching defs are computed for all regions.
    /// Returns the new reaching def at the op's exit.
    /// If values need to be merged at the start of a nested region, the input values must be added
    /// to `region_phis`. This will insert a memory phi at the beginning of the region's entry block,
    /// with the result value being the one added to `regions_to_process` in `setup_memory_ssa`.
    #[allow(
        clippy::too_many_arguments,
        reason = "packing them into structs would be more complex"
    )]
    fn finalize_memory_ssa(
        &self,
        ctx: &Context,
        state: &mut MemorySSAContext,
        entry_reaching_def: MemoryValue,
        has_memory_defs: bool,
        reaching_at_region_entry: &HMap<Ptr<Region>, MemoryValue>,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, MemoryValue>,
        region_phis: &mut SmallMap<Ptr<Region>, RegionMemoryPhiInputs, 2>,
    ) -> RegionMemoryValue;
}

/// Describes a type that can be broken down into indexable sub-element types.
#[type_interface]
pub trait DestructurableTypeInterface {
    verify_ty_succ!();

    /// Destructures the type into subelements into a map of indices to
    /// types of subelements. Returns nothing if the type cannot be destructured.
    fn subelement_index_map(&self, ctx: &Context) -> Option<HMap<AttrObj, TypeHandle>>;

    /// Indicates which type is held at the provided index, returning None
    /// if no type could be computed. While this can return information
    /// even when the type cannot be completely destructured, it must be coherent
    /// with the types returned by `subelement_index_map` when they exist.
    fn type_at_index(&self, ctx: &Context, index: &AttrObj) -> TypeHandle;
}

/// Describes operations creating values of aggregates that can be
/// destructured into multiple smaller values.
#[op_interface]
pub trait DestructurableConstructorOpInterface {
    verify_op_succ!();

    /// Returns the list of value for which destructuring should be attempted,
    /// specifying in which way the value should be destructured into subvalues.
    /// This computes the type of the value for each subvalue to be generated. The type of the value
    /// must implement [`DestructurableTypeInterface`].
    ///
    /// No IR mutation is allowed in this method.
    fn destructurable_values(&self, ctx: &Context) -> Vec<DestructurableValueSlot>;

    /// Destructures this value into multiple subvalues. The original value must still exist
    /// at the end of this call. Only generates subvalues for the indices found in
    /// `used_indices` since all other subvalues are unused.
    ///
    /// The rewriter is located before this op.
    fn destructure(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        used_indices: &SmallSet<AttrObj, 8>,
        rewriter: &mut PassRewriter,
        new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
    ) -> HMap<AttrObj, ValueSlot>;

    /// Hook triggered once the destructuring of a value is complete, meaning the
    /// original value is no longer being referred to and could be deleted.
    /// This will only be called for values declared by this operation.
    ///
    /// Must return a new destructurable constructor op if this hook creates
    /// a new destructurable op, `None` otherwise.
    fn handle_destructuring_complete(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        rewriter: &mut PassRewriter,
    ) -> Option<TraitOp<dyn DestructurableConstructorOpInterface>>;
}

/// Describes operations that can access a sub-element of a destructurable value.
#[op_interface]
pub trait DestructurableAccessorOpInterface {
    verify_op_succ!();

    /// For a given destructurable value, returns whether this operation can
    /// rewire its uses of the value to use the values generated after
    /// destructuring. This may involve creating new operations.
    ///
    /// This method must also register the indices it will access within the
    /// `used_indices` set. If the accessor generates new values mapping to
    /// subelements, they must be registered in `must_be_safely_used` to ensure
    /// they are used in a safe manner.
    ///
    /// No IR mutation is allowed in this method.
    fn can_rewire(
        &self,
        ctx: &Context,
        value: &DestructurableValueSlot,
        used_indices: &mut SmallSet<AttrObj, 8>,
        must_be_safely_used: &mut Vec<ValueSlot>,
    ) -> bool;

    /// Rewires the use of a slot to the generated subvalues, without deleting
    /// any operation. Returns whether the accessor should be deleted.
    ///
    /// Deletion of operations is not allowed, only the accessor can be
    /// scheduled for deletion by returning the appropriate value.
    fn rewire(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        subvalues: &HMap<AttrObj, ValueSlot>,
        rewriter: &mut PassRewriter,
    ) -> DeletionKind;
}

#[op_interface]
pub trait SafeMemorySlotAccessOpInterface {
    verify_op_succ!();

    #[allow(clippy::result_unit_err)]
    /// Returns whether all accesses in this operation to the provided value are
    /// done in a safe manner. To be safe, the access must only access the value
    /// inside the bounds that its type implies.
    ///
    /// If the safety of the accesses depends on the safety of the accesses to
    /// further value, the result of this method will be conditioned to
    /// the safety of the accesses to the value added by this method to
    /// `must_be_safely_used`.
    ///
    /// No IR mutation is allowed in this method.
    fn ensure_only_safe_accesses(
        &self,
        ctx: &Context,
        value: &ValueSlot,
        must_be_safely_used: &mut Vec<ValueSlot>,
    ) -> LogicalResult;
}

#[derive(new, Debug)]
pub struct ValueSlot {
    pub value: Value,
    pub elem_ty: TypeHandle,
}

#[derive(Debug)]
pub struct DestructurableValueSlot {
    pub slot: ValueSlot,
    pub subelement_types: HMap<AttrObj, TypeHandle>,
}
