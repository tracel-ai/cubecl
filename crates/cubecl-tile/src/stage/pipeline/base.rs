//! The buffered walk ([`Stages`]): `depth` [`Slot`]s driven as a circular software pipeline,
//! `depth - 1` regions in flight while one computes; built, filled and consumed here, driven by
//! the [`schedule`](super::schedule).

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::payload::base::{Payload, PayloadExpand, StageSpec};
use super::payload::pair::OperandPair;
use super::plan::StagePlan;
use super::schedule::Prefetch;
use crate::*;

/// What a plane does with the stages' slots.
#[derive(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants)]
pub enum Role {
    /// Fills the slots and takes no tile.
    Fill,
    /// Reads the slots and computes.
    Compute,
}

/// The `depth` slots of one buffered walk and the operands they are filled from.
#[derive(CubeType)]
pub struct Stages<T: CubeType> {
    pub(crate) slots: Sequence<Slot<T>>,
    pub(crate) sources: T,
    #[allow(dead_code)]
    #[cube(comptime)]
    pub(crate) depth: usize,
    /// Planes that only fill these slots.
    #[cube(comptime)]
    pub(crate) fillers: usize,
}

#[cube]
impl<T: CubeType> Stages<T> {
    /// Wrap already-built slots over the operands they stage.
    pub(crate) fn wrap(
        slots: Sequence<Slot<T>>,
        sources: T,
        #[comptime] depth: usize,
        #[comptime] fillers: usize,
    ) -> Stages<T> {
        comptime!(assert!(
            depth > 0,
            "Stages: a pipeline needs at least one slot"
        ));
        Stages::<T> {
            slots,
            sources,
            depth,
            fillers,
        }
    }

    /// What this plane does with the stages' slots; filler planes sit at the end of the cube.
    pub fn role(&self) -> Role {
        if comptime!(self.fillers == 0) {
            Role::new_Compute()
        } else if UNIT_POS >= Meeting::consumers(comptime!(self.fillers)) {
            Role::new_Fill()
        } else {
            Role::new_Compute()
        }
    }

    /// Slot `index`, known at comptime.
    pub fn slot_mut(&mut self, #[comptime] index: usize) -> &mut Slot<T> {
        self.slots.index_mut(index)
    }
}

#[cube]
impl<P: Payload<P> + Clone + CubeType<ExpandType: Clone>> Stages<P> {
    /// Build the `depth` slots staging `sources` into shared memory laid out as `storage`.
    ///
    /// `width` overrides the stage's line width; the operand must then be scalar and unpacked.
    /// `decoded` is the element a stage whose fill decodes its operand holds, `f32` where `None`.
    #[allow(dead_code)]
    pub(crate) fn new(
        walk: &Walk,
        sources: &P,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] depth: usize,
        #[comptime] decoded: Option<ElemType>,
    ) -> Stages<P> {
        let operands = sources.operands();
        let owner = walk.stage_owner();
        let plan = comptime!(StagePlan::new(&operands, &walk.space, &walk.level, owner));
        let spec = comptime!(StageSpec {
            level: walk.level.clone(),
            depth: walk.depth(),
            storage,
            width,
            decoded,
            owner: plan.owner(),
        });

        // Later slots share buffers the first slot owns, so it is built first.
        let first = sources.staged(
            sources,
            comptime!(spec.clone()),
            comptime!(plan.refills(FIRST_SLOT)),
        );
        let mut slots = Sequence::<Slot<P>>::new();
        #[unroll]
        for slot in 0..depth {
            let data = if comptime!(slot == FIRST_SLOT) {
                first.clone()
            } else {
                sources.staged(
                    &first,
                    comptime!(spec.clone()),
                    comptime!(plan.refills(slot)),
                )
            };
            slots.push(Slot::wrap(
                data,
                Meeting::new(
                    comptime!(plan.sync()),
                    comptime!(plan.collective_full()),
                    comptime!(plan.commits()),
                    comptime!(plan.fences()),
                    comptime!(plan.fillers()),
                ),
                comptime!(plan.refills(slot)),
            ));
        }
        Stages::wrap(slots, sources.clone(), depth, comptime!(plan.fillers()))
    }

    /// Free the shared memory every slot holds, for what the kernel declares after it to take:
    /// once the walk is over, nothing reads a stage again. Every unit of the cube reaches it, and
    /// a cube barrier comes before any unit touches what is declared in its place, as
    /// [`Tile::smem_accumulation`] and [`Tile::smem_cyclic_accumulation`] open with.
    pub fn free(&self) {
        #[unroll]
        for slot in 0..comptime!(self.depth) {
            self.slots.index(slot).data.free();
        }
    }

    /// Fill slot `slot` from `region`'s window: one step of a [`Fill`](Role::Fill) plane's walk.
    pub fn fill(&mut self, #[comptime] slot: usize, region: &Region) {
        let sources = self.sources.clone();
        self.slot_mut(slot)
            .refill(&sources, region, comptime!(Refill::EveryRegion));
    }
}

#[cube]
impl Stages<OperandPair> {
    /// The `depth` slots staging both operands of a contraction into shared memory laid out as
    /// `storage`.
    ///
    /// A stage whose fill decodes its operand, from scales, a codebook or fields as wide as their
    /// word, holds the values at `decoded`, or at `f32` where `None`. An operand staged as it lies
    /// keeps its own element.
    pub fn smem(
        walk: &Walk,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
        #[comptime] decoded: Option<ElemType>,
    ) -> Stages<OperandPair> {
        let sources = OperandPair {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        };
        Stages::new(walk, &sources, storage, comptime!(None), depth, decoded)
    }
}

#[cube]
impl Stages<Tile> {
    /// [`smem`](Stages::smem) for the sole operand `input`.
    pub fn smem_single(
        walk: &Walk,
        input: &Tile,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
        #[comptime] decoded: Option<ElemType>,
    ) -> Stages<Tile> {
        Stages::new(walk, input, storage, comptime!(None), depth, decoded)
    }

    /// [`smem_single`](Stages::smem_single) with the stage served in `width`-wide lines.
    pub fn smem_single_at(
        walk: &Walk,
        input: &Tile,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] depth: usize,
        #[comptime] decoded: Option<ElemType>,
    ) -> Stages<Tile> {
        Stages::new(walk, input, storage, width, depth, decoded)
    }
}

// Line widths of a register-staged fill, bound by `fetch_buffers`.
define_size!(pub(crate) FL);
define_size!(pub(crate) FR);

/// Registers holding one unit's register-staged fill of both operands.
pub(crate) type FetchBuffers<Lhs, Rhs> = (Array<Vector<Lhs, FL>>, Array<Vector<Rhs, FR>>);

/// Bind the fetched line widths `FL` and `FR` for the rest of the kernel's scope.
#[cube]
fn register_fetched_widths(#[comptime] lhs: usize, #[comptime] rhs: usize) {
    intrinsic!(|scope| {
        scope.register_size::<FL>(lhs);
        scope.register_size::<FR>(rhs);
    });
}

#[cube]
impl Stages<OperandPair> {
    /// Registers holding one unit's fill of both operands across a contraction.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetch_buffers<Lhs: Numeric, Rhs: Numeric>(&self) -> FetchBuffers<Lhs, Rhs> {
        let staged = self.slots.index(FIRST_SLOT);
        let lhs = staged.data.lhs.fetched_scalars();
        let rhs = staged.data.rhs.fetched_scalars();
        comptime!(assert!(
            Prefetch::fits(lhs + rhs),
            "Stages: a fill fetched into registers holds {lhs} + {rhs} scalars a unit, past the \
             {MOST_FETCHED_SCALARS} a unit keeps beside its contraction"
        ));
        let lhs_width = staged.data.lhs.vector_size();
        let rhs_width = staged.data.rhs.vector_size();
        register_fetched_widths(lhs_width, rhs_width);
        (
            Array::<Vector<Lhs, FL>>::new(comptime!(lhs / lhs_width)),
            Array::<Vector<Rhs, FR>>::new(comptime!(rhs / rhs_width)),
        )
    }

    /// Read this unit's share of filling slot `slot` with `sources` at `region` into `lhs` and
    /// `rhs`.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetch<Lhs: Numeric, Rhs: Numeric>(
        &self,
        #[comptime] slot: usize,
        sources: &OperandPair,
        region: &Region,
        lhs: &mut Array<Vector<Lhs, FL>>,
        rhs: &mut Array<Vector<Rhs, FR>>,
    ) {
        let staged = self.slots.index(slot);
        staged.data.lhs.fetch_from(&sources.lhs.at(region), lhs);
        staged.data.rhs.fetch_from(&sources.rhs.at(region), rhs);
    }

    /// Write what [`fetch`](Stages::fetch) read into slot `slot`; the caller owns the rendezvous.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn store<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        #[comptime] slot: usize,
        lhs: &Array<Vector<Lhs, FL>>,
        rhs: &Array<Vector<Rhs, FR>>,
    ) {
        let slot = self.slot_mut(slot);
        slot.data.lhs.store_fetched(lhs);
        slot.data.rhs.store_fetched(rhs);
    }

    /// Refuse stages a register-staged schedule cannot drive.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn assert_copied_by_every_unit(&self) {
        let lhs = self.sources.lhs.delivery();
        let rhs = self.sources.rhs.delivery();
        comptime!(assert!(
            self.fillers == 0 && lhs == Delivery::SyncPerUnit && rhs == Delivery::SyncPerUnit,
            "Stages: a register-staged schedule fills its slots with every unit's own copy of \
             memory; these stages are filled by {} plane(s) of their own, or their sources are \
             delivered {lhs:?} and {rhs:?}",
            self.fillers
        ));
    }
}

// `consume` is spelled per payload shape: closure inference cannot resolve `P::ExpandType`.
impl Slot<OperandPair> {
    /// Wait for the slot's fill, pass its two tiles to `compute`, then free the slot.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile, &Tile)) {
        unexpanded!()
    }
}

impl SlotExpand<OperandPair> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand, &TileExpand),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data.lhs, &self.data.rhs);
        self.__expand_release_read_method(scope);
    }
}

impl Slot<Tile> {
    /// [`consume`](Slot::consume) for the sole operand.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile)) {
        unexpanded!()
    }
}

impl SlotExpand<Tile> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data);
        self.__expand_release_read_method(scope);
    }
}

impl Stages<OperandPair> {
    /// Consume slot `slot`: one step of a [`Compute`](Role::Compute) plane's walk.
    pub fn consume(&mut self, _slot: usize, _compute: impl FnOnce(&Tile, &Tile)) {
        unexpanded!()
    }
}

impl StagesExpand<OperandPair> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, slot: usize, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand, &TileExpand),
    {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_consume_method(scope, compute);
    }
}

impl Stages<Tile> {
    /// [`consume`](Stages::consume) for the sole operand.
    pub fn consume(&mut self, _slot: usize, _compute: impl FnOnce(&Tile)) {
        unexpanded!()
    }
}

impl StagesExpand<Tile> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, slot: usize, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand),
    {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_consume_method(scope, compute);
    }
}
