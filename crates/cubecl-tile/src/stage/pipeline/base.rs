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

        // Every slot's barriers are declared before any slot's data: the barriers stay in use to
        // the end, while the data is what `free` hands to what the kernel declares after the walk,
        // which takes it only as one run.
        let mut meetings = Sequence::<Meeting>::new();
        #[unroll]
        for _ in 0..depth {
            meetings.push(Meeting::new(
                plan.sync(),
                plan.collective_full(),
                plan.commits(),
                plan.fences(),
                plan.fillers(),
            ));
        }
        // Later slots share buffers the first slot owns, so it is built first.
        let first = sources.staged(sources, spec.clone(), comptime!(plan.refills(FIRST_SLOT)));
        let mut slots = Sequence::<Slot<P>>::new();
        #[unroll]
        for slot in 0..depth {
            let data = if comptime!(slot == FIRST_SLOT) {
                first.clone()
            } else {
                sources.staged(&first, spec.clone(), plan.refills(slot))
            };
            slots.push(Slot::wrap(
                data,
                meetings.index(slot).clone(),
                plan.refills(slot),
            ));
        }
        Stages::wrap(slots, sources.clone(), depth, plan.fillers())
    }

    /// Free the shared memory every slot holds, for what the kernel declares after it to take:
    /// once the walk is over, nothing reads a stage again. Every unit of the cube reaches it, and
    /// a cube barrier comes before any unit touches what is declared in its place, as
    /// [`Tile::smem_accumulation`] and [`Tile::smem_cyclic_accumulation`] open with. The slots'
    /// rendezvous end with them, so a ring built again in their place, in a loop, meets on new
    /// barriers.
    pub fn free(&self) {
        #[unroll]
        for slot in 0..comptime!(self.depth) {
            self.slots.index(slot).meeting.end();
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
        let staged = comptime!(StagedOperands::Both);
        Stages::smem_staging(walk, lhs, rhs, staged, storage, depth, decoded)
    }

    /// [`smem`](Stages::smem) staging the operands `staged` names, the other read where it lies.
    pub fn smem_staging(
        walk: &Walk,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] staged: StagedOperands,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
        #[comptime] decoded: Option<ElemType>,
    ) -> Stages<OperandPair> {
        let sources = OperandPair {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            staged,
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

/// Registers holding one unit's register-staged fill of both operands: each operand's lines, and
/// the scales of the lines a decoded operand lands (one unused scale for a straight one).
pub(crate) type FetchBuffers<Lhs, Rhs> = (
    Array<Vector<Lhs, FL>>,
    Array<Vector<Rhs, FR>>,
    Array<u32>,
    Array<u32>,
);

/// How a register fetch holds one operand's fill between the fetch and the store.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FetchedFill {
    /// The source's lines as they lie, stored as they were read.
    Straight,
    /// The packed source's loads as stored and the scale of every line they land, decoded at the
    /// store ([`Tile::fetch_decoded_from`]).
    Decoded,
}

/// Bind the fetched line widths `FL` and `FR` for the rest of the kernel's scope.
#[cube]
fn register_fetched_widths(#[comptime] lhs: usize, #[comptime] rhs: usize) {
    intrinsic!(|scope| {
        scope.register_size::<FL>(lhs);
        scope.register_size::<FR>(rhs);
    });
}

#[cube]
impl Tile {
    /// How a register fetch holds this stage's fill from `src`, unless it cannot.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn register_fetch(&self, src: &Tile) -> comptime_type!(Option<FetchedFill>) {
        let straight = self.fetches_straight_from(src);
        let staged = self.packing();
        let decodes = src.decodes_into(staged);
        let turns = self.decoded_fetch_turns(src);
        comptime!(match (straight, decodes && turns.is_some()) {
            (true, _) => Some(FetchedFill::Straight),
            (false, true) => Some(FetchedFill::Decoded),
            (false, false) => None,
        })
    }

    /// 32-bit registers one unit holds between the fetch and the store of this stage's fill from
    /// `src`, counted as [`Prefetch`] counts them for a caller that plans before expanding.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn register_fetched_registers(&self, src: &Tile) -> comptime_type!(Option<usize>) {
        let fill = self.register_fetch(src);
        let straight = self.fetched_scalars(src);
        let width = self.fetched_width(src);
        let decoded = self.decoded_fetched_registers(src);
        comptime!(match fill {
            Some(FetchedFill::Straight) => straight.map(|scalars| {
                Prefetch::straight_registers(scalars / width, width, src.element.size_bits())
            }),
            Some(FetchedFill::Decoded) => decoded,
            None => None,
        })
    }

    /// The element a register fetch holds this stage's fill from `src` at: the source's own, or
    /// the words a decoded one is stored in.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetched_elements`.
    pub(crate) fn register_fetched_element(&self, src: &Tile) -> comptime_type!(ElemType) {
        let fill = self.register_fetch(src);
        comptime!(match fill {
            Some(FetchedFill::Decoded) => ElemType::UInt(cubecl::ir::UIntKind::U32),
            Some(FetchedFill::Straight) | None => src.element,
        })
    }

    /// The width of a register fetch's lines, and its count of lines and of scales, for this
    /// stage's fill from `src`.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch_buffers`.
    fn register_fetched_shape(&self, src: &Tile) -> comptime_type!((usize, usize, usize)) {
        let fill = self.register_fetch(src);
        let straight = self.fetched_scalars(src);
        let straight_width = self.fetched_width(src);
        let turns = self.decoded_fetch_turns(src);
        let decoded_width = self.decoded_fetch_width(src);
        let per_load = self.decoded_scales_per_load(src);
        comptime!(match fill {
            Some(FetchedFill::Straight) => {
                let scalars = straight.expect("Stages: a register-staged fill has a known share");
                (straight_width, scalars / straight_width, 1)
            }
            Some(FetchedFill::Decoded) => {
                let turns = turns.expect("Stages: a register-staged fill has a known share");
                (decoded_width, turns, turns * per_load)
            }
            None => panic!("Stages: a register-staged fill on a stage that cannot fetch"),
        })
    }

    /// Read this unit's share of this stage's fill from `src` into `lines` and `scales`.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch`.
    fn register_fetch_from<T: Numeric, W: Size>(
        &self,
        src: &Tile,
        lines: &mut Array<Vector<T, W>>,
        scales: &mut Array<u32>,
    ) {
        let fill = self.register_fetch(src);
        match comptime!(fill) {
            Some(FetchedFill::Decoded) => self.fetch_decoded_from(src, lines, scales),
            Some(FetchedFill::Straight) | None => self.fetch_from(src, lines),
        }
    }

    /// Write what [`register_fetch_from`](Tile::register_fetch_from) read of `src` into this stage.
    #[allow(dead_code)] // Reached through its expand, from `Stages::store`.
    fn register_store<T: Numeric, W: Size>(
        &mut self,
        src: &Tile,
        lines: &Array<Vector<T, W>>,
        scales: &Array<u32>,
    ) {
        let fill = self.register_fetch(src);
        match comptime!(fill) {
            Some(FetchedFill::Decoded) => self.store_decoded(src, lines, scales),
            Some(FetchedFill::Straight) | None => self.store_fetched(lines),
        }
    }
}

#[cube]
impl Stages<OperandPair> {
    /// Whether `prefetched` holds the next stage in registers; false in the
    /// cases [`Prefetch::InRegisters`] lists.
    pub fn prefetches_in_registers(&self) -> comptime_type!(bool) {
        let fills = self.fills_through_registers();
        comptime!(fills && self.depth <= 2)
    }

    /// Whether every unit can hold its share of both operands' fill of a slot in registers
    /// between the fill's loads and its stores: the cases [`Prefetch::InRegisters`] lists, but
    /// for the depth.
    pub(crate) fn fills_through_registers(&self) -> comptime_type!(bool) {
        let staged = self.slots.index(FIRST_SLOT);
        let operands = comptime!(staged.data.staged);
        let fixed = staged.has_fixed();
        let lhs_beside = self.sources.lhs.scales_beside();
        let rhs_beside = self.sources.rhs.scales_beside();
        let lhs_delivery = self.sources.lhs.delivery();
        let rhs_delivery = self.sources.rhs.delivery();
        let lhs_registers = staged
            .data
            .lhs
            .register_fetched_registers(&self.sources.lhs);
        let rhs_registers = staged
            .data
            .rhs
            .register_fetched_registers(&self.sources.rhs);
        comptime!(
            operands == StagedOperands::Both
                && !fixed
                && !lhs_beside
                && !rhs_beside
                && self.fillers == 0
                && lhs_delivery.holds(Prefetch::InRegisters).is_ok()
                && rhs_delivery.holds(Prefetch::InRegisters).is_ok()
                && matches!(
                    (lhs_registers, rhs_registers),
                    (Some(l), Some(r)) if Prefetch::fits_registers(l + r)
                )
        )
    }

    /// The elements the registers of a register-staged fill hold each operand at.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetched_elements(&self) -> comptime_type!((ElemType, ElemType)) {
        let staged = self.slots.index(FIRST_SLOT);
        let lhs = staged.data.lhs.register_fetched_element(&self.sources.lhs);
        let rhs = staged.data.rhs.register_fetched_element(&self.sources.rhs);
        comptime!((lhs, rhs))
    }

    /// Registers holding one unit's fill of both operands across a contraction; only where
    /// [`prefetches_in_registers`](Stages::prefetches_in_registers) holds.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetch_buffers<Lhs: Numeric, Rhs: Numeric>(&self) -> FetchBuffers<Lhs, Rhs> {
        let staged = self.slots.index(FIRST_SLOT);
        let lhs_shape = staged.data.lhs.register_fetched_shape(&self.sources.lhs);
        let rhs_shape = staged.data.rhs.register_fetched_shape(&self.sources.rhs);
        register_fetched_widths(comptime!(lhs_shape.0), comptime!(rhs_shape.0));
        (
            Array::<Vector<Lhs, FL>>::new(comptime!(lhs_shape.1)),
            Array::<Vector<Rhs, FR>>::new(comptime!(rhs_shape.1)),
            Array::<u32>::new(comptime!(lhs_shape.2)),
            Array::<u32>::new(comptime!(rhs_shape.2)),
        )
    }

    /// Read this unit's share of filling slot `slot` with `sources` at `region` into the buffers.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fetch<Lhs: Numeric, Rhs: Numeric>(
        &self,
        #[comptime] slot: usize,
        sources: &OperandPair,
        region: &Region,
        lhs: &mut Array<Vector<Lhs, FL>>,
        rhs: &mut Array<Vector<Rhs, FR>>,
        lhs_scales: &mut Array<u32>,
        rhs_scales: &mut Array<u32>,
    ) {
        let staged = self.slots.index(slot);
        staged
            .data
            .lhs
            .register_fetch_from(&sources.lhs.at(region), lhs, lhs_scales);
        staged
            .data
            .rhs
            .register_fetch_from(&sources.rhs.at(region), rhs, rhs_scales);
    }

    /// Write what [`fetch`](Stages::fetch) read of `sources` at `region` into slot `slot`; the
    /// caller owns the rendezvous.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        #[comptime] slot: usize,
        sources: &OperandPair,
        region: &Region,
        lhs: &Array<Vector<Lhs, FL>>,
        rhs: &Array<Vector<Rhs, FR>>,
        lhs_scales: &Array<u32>,
        rhs_scales: &Array<u32>,
    ) {
        let slot = self.slot_mut(slot);
        slot.data
            .lhs
            .register_store(&sources.lhs.at(region), lhs, lhs_scales);
        slot.data
            .rhs
            .register_store(&sources.rhs.at(region), rhs, rhs_scales);
    }

    /// Fill slot `slot` from `sources` at `region` through registers: every unit issues its
    /// loads of both operands, waits for the slot to be free, then stores them, so the loads'
    /// round trips overlap one another and the wait rather than following each other. Only
    /// where [`fills_through_registers`](Stages::fills_through_registers) holds.
    #[allow(dead_code)] // Reached through its expand, from the slot schedule.
    pub(crate) fn fill_through_registers<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        #[comptime] slot: usize,
        sources: &OperandPair,
        region: &Region,
    ) {
        let (mut lhs, mut rhs, mut lhs_scales, mut rhs_scales) =
            self.fetch_buffers::<Lhs, Rhs>();
        self.fetch::<Lhs, Rhs>(
            slot,
            sources,
            region,
            &mut lhs,
            &mut rhs,
            &mut lhs_scales,
            &mut rhs_scales,
        );
        // The loads stay ahead of the wait for the slot, and that wait ahead of the stores and
        // the work between loads and stores (a transpose): a scheduler would otherwise sink the
        // later loads past the first ones' stores, and the wait past the loads' arrival.
        schedule_barrier();
        self.slot_mut(slot).meeting.acquire_write();
        schedule_barrier();
        self.store::<Lhs, Rhs>(
            slot,
            sources,
            region,
            &lhs,
            &rhs,
            &lhs_scales,
            &rhs_scales,
        );
        self.slot_mut(slot).meeting.release_write();
    }
}

// `consume` is spelled per payload shape, as [`Slot::consume`] is.
impl Stages<OperandPair> {
    /// Consume slot `slot`, filled for `region`: one step of a [`Compute`](Role::Compute) plane's
    /// walk.
    pub fn consume(&mut self, _slot: usize, _region: &Region, _compute: impl FnOnce(&Tile, &Tile)) {
        unexpanded!()
    }
}

impl StagesExpand<OperandPair> {
    pub fn __expand_consume_method<F>(
        &mut self,
        scope: &Scope,
        slot: usize,
        region: &RegionExpand,
        compute: F,
    ) where
        F: FnOnce(&Scope, &TileExpand, &TileExpand),
    {
        self.placed_slot(scope, slot, region)
            .__expand_consume_method(scope, compute);
    }
}

impl Stages<Tile> {
    /// [`consume`](Stages::consume) for the sole operand.
    pub fn consume(&mut self, _slot: usize, _region: &Region, _compute: impl FnOnce(&Tile)) {
        unexpanded!()
    }
}

impl StagesExpand<Tile> {
    pub fn __expand_consume_method<F>(
        &mut self,
        scope: &Scope,
        slot: usize,
        region: &RegionExpand,
        compute: F,
    ) where
        F: FnOnce(&Scope, &TileExpand),
    {
        self.placed_slot(scope, slot, region)
            .__expand_consume_method(scope, compute);
    }
}
