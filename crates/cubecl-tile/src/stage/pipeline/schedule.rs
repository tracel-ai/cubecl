//! The schedules that drive a walk over [`Stages`]: fill each slot ahead of the region that
//! reads it.

use cubecl::frontend::branch::{if_else_expand, if_expand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::payload::base::{Payload, PayloadExpand};
use crate::*;

/// How stages fill their slots from their own sources, at expand level.
pub trait StagesFill {
    /// Whether any operand's window is fixed across the walk.
    fn has_fixed(&self, scope: &Scope) -> bool;
    /// Fill slot `slot`'s fixed operands from `region`'s window.
    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Fill slot `slot`'s streamed operands from `region`'s window.
    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// [`fill_streamed`](StagesFill::fill_streamed) for a region the walk proves inside the runs
    /// its operands hold ([`Payload::inside_runs`]).
    fn fill_inside_run(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Publish slot `slot`'s last fill where no later fill will (see [`Slot::publish`]).
    fn publish(&mut self, scope: &Scope, slot: usize);
    /// Place what slot `slot` reads where it lies at `region`, which it is about to be read for.
    fn place(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
}

impl<T: CubeType> Stages<T> {
    /// Walk `walk`'s regions, filling slots from the stages' sources and `compute` consuming each
    /// through [`consume`](Slot::consume).
    pub fn pipelined<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<T>, &Region),
    {
        unexpanded!()
    }
}

impl<T: CubeType> StagesExpand<T>
where
    StagesExpand<T>: StagesFill,
{
    pub fn __expand_pipelined_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        shared::run(
            scope,
            walk,
            self,
            |scope, stages, slot, region, reach| match reach {
                Reach::Walked => stages.fill_streamed(scope, slot, region),
                Reach::InsideRuns => stages.fill_inside_run(scope, slot, region),
            },
            compute,
        )
    }
}

/// Which sources a schedule's fill reads: the stages' own, or the ones a walk proves its region
/// inside the runs of ([`StagesFill::fill_inside_run`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    Walked,
    InsideRuns,
}

/// Where a walk's next stage waits while the previous one is contracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefetch {
    /// In its own slot, filled before the previous contraction.
    InSlots,
    /// In registers, loaded before the contraction and stored after. A fill that copies its
    /// source's lines as they lie holds them; a fill that decodes a packed source holds its words
    /// and the word of the block scale each line lands under, and decodes at the store, where its
    /// block scales are packed words and every coarser scale is one value across the stage. Takes
    /// [`InSlots`](Prefetch::InSlots) where the registers cannot hold the fill: fixed operands,
    /// scales kept beside their values, filler planes, a source an engine moves, a depth past 2, a
    /// fill that converts, rewidens, gathers or packs its source or masks its stage, a
    /// decoding fill whose scales are not held as words, a stage whose per-unit share is unknown,
    /// and loads past [`fits_registers`](Prefetch::fits_registers).
    InRegisters,
}

impl Prefetch {
    /// Loads one of `units` units takes of a stage of `elements` values read `width` at a time.
    ///
    /// # Assumptions
    ///
    /// - `units` is a known count, above zero.
    ///
    /// # Panics
    ///
    /// - Where `units` is zero, which deals the lines to no known count of units.
    pub fn turns(elements: usize, units: usize, width: usize) -> usize {
        Level::copied_by_cube(units)
            .copy_turns(elements.div_ceil(width))
            .expect(
                "Prefetch::turns: a register prefetch should deal its lines to a known count of \
                 units",
            )
    }

    /// 32-bit registers a unit holds of a fill copying its source's lines as they lie: `turns`
    /// loads of `width` values of `bits` bits each.
    pub fn straight_registers(turns: usize, width: usize, bits: usize) -> usize {
        (turns * width * bits).div_ceil(u32::BITS as usize)
    }

    /// 32-bit registers a unit holds of a fill decoding a packed source: each of `turns` loads of
    /// `width` values of `bits` bits as the words it is stored in, and the word of the block scale
    /// each of the lines of `line` values it lands is decoded under.
    pub fn decoded_registers(turns: usize, width: usize, bits: usize, line: usize) -> usize {
        turns * ((width * bits).div_ceil(u32::BITS as usize) + width / line)
    }

    /// Whether a register prefetch holding `registers` 32-bit registers a unit, over both
    /// operands, is accepted.
    ///
    /// # Assumptions
    ///
    /// - `registers` counts the lines a unit fetches, not the addresses it keeps across the
    ///   contraction.
    ///
    /// # Guarantees
    ///
    /// - Accepts at most a quarter of the 256 registers a unit addresses.
    pub fn fits_registers(registers: usize) -> bool {
        registers <= MOST_FETCHED_REGISTERS
    }
}

impl Stages<OperandPair> {
    /// Walk `walk`'s stages with the schedule `prefetch` selects.
    pub fn walk<F>(&mut self, _walk: Walk, _prefetch: Prefetch, _compute: F)
    where
        F: FnMut(&mut Slot<OperandPair>, &Region),
    {
        unexpanded!()
    }
}

impl StagesExpand<OperandPair> {
    pub fn __expand_walk_method<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        prefetch: Prefetch,
        compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        match prefetch {
            Prefetch::InSlots => self.pipelined_through_registers(scope, walk, compute),
            Prefetch::InRegisters => self.__expand_prefetched_method(scope, walk, compute),
        }
    }

    /// [`pipelined`](Stages::pipelined), each slot filled through registers where every unit can
    /// hold its share of both operands' fill ([`Stages::fills_through_registers`]): all of a
    /// unit's loads of a slot issue before its first store.
    fn pipelined_through_registers<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        if !self.__expand_fills_through_registers_method(scope) {
            return self.__expand_pipelined_method(scope, walk, compute);
        }
        define_scalar!(Lhs);
        define_scalar!(Rhs);
        let (lhs_element, rhs_element) = self.__expand_fetched_elements_method(scope);
        let _lhs = scope.define_type::<Lhs>(lhs_element);
        let _rhs = scope.define_type::<Rhs>(rhs_element);
        shared::run(
            scope,
            walk,
            self,
            |scope, stages, slot, region, reach| {
                let sources = match reach {
                    Reach::Walked => stages.sources.clone(),
                    Reach::InsideRuns => stages.sources.__expand_inside_runs_method(scope),
                };
                stages.__expand_fill_through_registers_method::<Lhs, Rhs>(
                    scope, slot, &sources, region,
                );
            },
            compute,
        )
    }
}

impl StagesExpand<OperandPair> {
    /// [`pipelined`](Stages::pipelined) with the next region's loads held in registers across the
    /// current contraction: [`Prefetch::InRegisters`].
    ///
    /// Falls back to [`pipelined`](Stages::pipelined) in the cases
    /// [`Prefetch::InRegisters`] lists.
    pub(crate) fn __expand_prefetched_method<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        if !self.__expand_prefetches_in_registers_method(scope) {
            return self.__expand_pipelined_method(scope, walk, compute);
        }
        let total = walk.__expand_total_method(scope);
        self.prime(scope, &walk, &total);
        self.prefetching_laps(scope, walk, total, compute);
    }

    /// Fill and publish the first region before any contraction, through the registers the laps
    /// fetch into, so its loads issue together as theirs do.
    fn prime(&mut self, scope: &Scope, walk: &WalkExpand, total: &NativeExpand<usize>) {
        define_scalar!(Lhs);
        define_scalar!(Rhs);
        let (lhs_element, rhs_element) = self.__expand_fetched_elements_method(scope);
        let _lhs = scope.define_type::<Lhs>(lhs_element);
        let _rhs = scope.define_type::<Rhs>(rhs_element);
        let any = 0usize.into_expand(scope).__expand_lt_method(scope, total);
        if_expand(scope, any, |scope| {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            let sources = self.sources.clone();
            self.__expand_fill_through_registers_method::<Lhs, Rhs>(
                scope, FIRST_SLOT, &sources, &first,
            );
            self.publish(scope, FIRST_SLOT);
        });
    }

    /// The laps whose every region has a next one to fetch, run with no branch around the fetch,
    /// so what it computes that no lap changes (each unit's place in the stage) leaves the loop.
    /// How many ran.
    #[allow(clippy::too_many_arguments)]
    fn steady_prefetching_laps<Lhs: Numeric, Rhs: Numeric, F>(
        &mut self,
        scope: &Scope,
        walk: &WalkExpand,
        total: NativeExpand<usize>,
        lhs: &mut <Array<Vector<Lhs, super::base::FL>> as CubeType>::ExpandType,
        rhs: &mut <Array<Vector<Rhs, super::base::FR>> as CubeType>::ExpandType,
        scales: &mut [<Array<u32> as CubeType>::ExpandType; 2],
        compute: &mut F,
    ) -> NativeExpand<usize>
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        // The laps whose every region has a next one to fetch: no branch around the fetch, so what
        // it computes that no lap changes (each unit's place in the stage) leaves the loop. Over a
        // run ([`Walk::run`]) they leave its last region to the last laps and fetch unchecked.
        let depth = self.depth;
        let (held_back, sources) = match walk.run {
            None => (0, self.sources.clone()),
            Some(_) => (1, self.sources.__expand_inside_runs_method(scope)),
        };
        let steady = steady_laps(scope, total, 1 + held_back, depth);
        let mut steady_body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_plus_method(scope, 1usize.into_expand(scope));
                let upcoming = walk.__expand_region_method(scope, next);
                let [lhs_scales, rhs_scales] = scales;
                self.__expand_fetch_method(
                    scope, target, &sources, &upcoming, lhs, rhs, lhs_scales, rhs_scales,
                );
                // The fetch's loads stay ahead of the contraction they are meant to hide behind:
                // a scheduler would otherwise sink each to its first use, after the contraction.
                schedule_barrier::expand(scope);
                let region = walk.__expand_region_method(scope, region_idx);
                let slot = self.placed_slot(scope, j, &region);
                compute(scope, slot, &region);
                // Nor does the store's decode rise into the contraction, where it would wait on
                // the loads the contraction hides.
                schedule_barrier::expand(scope);
                // One slot: every unit has read it before any overwrites it.
                if depth == 1 {
                    self.publish(scope, FIRST_SLOT);
                }
                self.__expand_store_method(
                    scope, target, &sources, &upcoming, lhs, rhs, lhs_scales, rhs_scales,
                );
                self.publish(scope, target);
            }
        };
        run_laps(
            scope,
            0usize.into_expand(scope),
            steady,
            walk.unroll,
            &mut steady_body,
        );
        steady
    }

    /// Each lap fetches the next region into registers, contracts, then stores into the freed slot.
    fn prefetching_laps<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        total: NativeExpand<usize>,
        mut compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        // Reached only past the guard in `prefetched`; a caller reaching it another way meets the
        // same rule here rather than a fetch that cannot hold its fill.
        assert!(
            self.__expand_prefetches_in_registers_method(scope),
            "Stages: a register-staged walk on stages that prefetch into their slots"
        );
        let depth = self.depth;
        // The registers a fetch holds a region in are at the elements the sources hold, or the
        // words a decoded source is stored in.
        define_scalar!(Lhs);
        define_scalar!(Rhs);
        let (lhs_element, rhs_element) = self.__expand_fetched_elements_method(scope);
        let _lhs = scope.define_type::<Lhs>(lhs_element);
        let _rhs = scope.define_type::<Rhs>(rhs_element);
        let (mut lhs, mut rhs, lhs_scales, rhs_scales) =
            self.__expand_fetch_buffers_method::<Lhs, Rhs>(scope);
        let mut scales = [lhs_scales, rhs_scales];
        let steady = self.steady_prefetching_laps(
            scope,
            &walk,
            total,
            &mut lhs,
            &mut rhs,
            &mut scales,
            &mut compute,
        );
        // The last laps, where the walk runs out under them.
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_plus_method(scope, 1usize.into_expand(scope));
                let in_walk = region_idx.__expand_lt_method(scope, &total);
                if_expand(scope, in_walk, |scope| {
                    let prefetching = next.__expand_lt_method(scope, &total);
                    let [lhs_scales, rhs_scales] = &mut scales;
                    if_expand(scope, prefetching, |scope| {
                        let upcoming = walk.__expand_region_method(scope, next);
                        let sources = self.sources.clone();
                        self.__expand_fetch_method(
                            scope, target, &sources, &upcoming, &mut lhs, &mut rhs, lhs_scales,
                            rhs_scales,
                        );
                    });
                    schedule_barrier::expand(scope);
                    let region = walk.__expand_region_method(scope, region_idx);
                    let slot = self.placed_slot(scope, j, &region);
                    compute(scope, slot, &region);
                    schedule_barrier::expand(scope);
                    if_expand(scope, prefetching, |scope| {
                        // One slot: every unit must finish reading before any overwrites it.
                        if depth == 1 {
                            self.publish(scope, FIRST_SLOT);
                        }
                        // The region the fetch read, named again: the decode of a decoded source
                        // reads its table and places its lines off it.
                        let upcoming = walk.__expand_region_method(scope, next);
                        let sources = self.sources.clone();
                        self.__expand_store_method(
                            scope, target, &sources, &upcoming, &lhs, &rhs, lhs_scales, rhs_scales,
                        );
                        self.publish(scope, target);
                    });
                });
            }
        };
        // One lap is left whenever the walk holds a region (`ceil(t / d) - floor((t - 1) / d)`),
        // none when it holds nothing, and the body guards every region it names: it runs once,
        // straight, rather than as a loop around a copy of the contraction. The region a run's
        // steady laps leave may make it two.
        match walk.run {
            None => body(scope, steady),
            Some(_) => {
                let laps = total
                    .__expand_plus_method(scope, (depth - 1).into_expand(scope))
                    .__expand_divided_by_method(scope, depth.into_expand(scope));
                run_laps(scope, steady, laps, walk.unroll, &mut body);
            }
        }
    }
}

/// Laps of `depth` regions each, out of a walk of `total`, in which every region has the one
/// `ahead` of it to fill: `(total - ahead) / depth`, none where the walk is shorter than `ahead`.
/// What a schedule runs with no branch around its fills before the laps where the walk runs out.
fn steady_laps(
    scope: &Scope,
    total: NativeExpand<usize>,
    ahead: usize,
    depth: usize,
) -> NativeExpand<usize> {
    total
        .__expand_max_with_method(scope, ahead.into_expand(scope))
        .__expand_minus_method(scope, ahead.into_expand(scope))
        .__expand_divided_by_method(scope, depth.into_expand(scope))
}

/// `body` on every lap in `from..to`, unrolled where the walk says.
fn run_laps(
    scope: &Scope,
    from: NativeExpand<usize>,
    to: NativeExpand<usize>,
    unroll: bool,
    body: &mut dyn FnMut(&Scope, NativeExpand<usize>),
) {
    let range = RangeExpand::new(from, to);
    if unroll {
        range.expand_unroll(scope, body);
    } else {
        range.expand(scope, body);
    }
}

impl<P: Payload<P> + CubeType<ExpandType: Clone>> StagesFill for StagesExpand<P> {
    fn has_fixed(&self, scope: &Scope) -> bool {
        self.slots
            .__expand_index_method(scope, FIRST_SLOT.into_expand(scope))
            .__expand_has_fixed_method(scope)
    }

    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_refill_method(scope, &sources, region, Refill::Once);
    }

    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_refill_method(scope, &sources, region, Refill::EveryRegion);
    }

    fn fill_inside_run(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.__expand_inside_runs_method(scope);
        self.__expand_slot_mut_method(scope, slot)
            .__expand_refill_method(scope, &sources, region, Refill::EveryRegion);
    }

    fn publish(&mut self, scope: &Scope, slot: usize) {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_publish_method(scope);
    }

    fn place(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.clone();
        let slot = self.__expand_slot_mut_method(scope, slot);
        let refills = slot.refills.clone();
        slot.data = slot
            .data
            .__expand_placed_at_method(scope, &sources, region, refills);
    }
}

impl<T: CubeType> StagesExpand<T>
where
    StagesExpand<T>: StagesFill,
{
    /// Slot `slot` as it is read for `region`.
    pub(crate) fn placed_slot(
        &mut self,
        scope: &Scope,
        slot: usize,
        region: &RegionExpand,
    ) -> &mut SlotExpand<T> {
        self.place(scope, slot, region);
        self.__expand_slot_mut_method(scope, slot)
    }
}

/// The shared schedule: prologue, laps prefetching one region ahead, drain publishing.
mod shared {
    use super::*;

    pub(super) fn run<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: WalkExpand,
        stages: &mut StagesExpand<T>,
        mut fill: Fill,
        mut compute: F,
    ) where
        StagesExpand<T>: StagesFill,
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand, Reach),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        let depth = stages.depth;
        let unroll = walk.unroll;
        let total = walk.__expand_total_method(scope);

        if stages.has_fixed(scope) {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            for slot in 0..depth {
                stages.fill_fixed(scope, slot, &first);
            }
        }

        for slot in 0..depth - 1 {
            let index = slot.into_expand(scope);
            let cond = index.__expand_lt_method(scope, &total);
            if_expand(scope, cond, |scope| {
                let region = walk.__expand_region_method(scope, slot.into_expand(scope));
                fill(scope, stages, slot, &region, Reach::Walked);
            });
        }

        let laps = total
            .__expand_plus_method(scope, (depth - 1).into_expand(scope))
            .__expand_divided_by_method(scope, depth.into_expand(scope));
        // Over a run ([`Walk::run`]) the steady laps leave its last region to the last laps and fill
        // unchecked.
        let (held_back, reach) = match walk.run {
            None => (0, Reach::Walked),
            Some(_) => (1, Reach::InsideRuns),
        };
        let steady = steady_ahead_laps(
            scope,
            &walk,
            stages,
            total,
            held_back,
            &mut |scope, stages: &mut StagesExpand<T>, slot, region: &RegionExpand| {
                fill(scope, stages, slot, region, reach)
            },
            &mut compute,
        );
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                if depth == 1 {
                    let region = walk.__expand_region_method(scope, region_idx);
                    fill(scope, stages, FIRST_SLOT, &region, Reach::Walked);
                    stages.publish(scope, FIRST_SLOT);
                    let slot = stages.placed_slot(scope, FIRST_SLOT, &region);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let in_walk = region_idx.__expand_lt_method(scope, &total);
                    // Compute is emitted once, after the fill branch, keeping its accumulator live.
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        fill(scope, stages, (j + depth - 1) % depth, &prefetch, Reach::Walked);
                    })
                    .or_else(scope, |scope| {
                        // Draining: no fill follows, so this consume publishes.
                        if_expand(scope, in_walk, |scope| {
                            stages.publish(scope, j);
                        });
                    });
                    if_expand(scope, in_walk, |scope| {
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = stages.placed_slot(scope, j, &region);
                        compute(scope, slot, &region);
                    });
                }
            }
        };
        run_laps(scope, steady, laps, unroll, &mut body);
    }

    /// The laps whose every region has the one `depth - 1` ahead to fill, `held_back` regions
    /// short of the walk's end, run with no branch around the fill, so what it computes that no
    /// lap changes (each unit's place in the stage) leaves the loop; the last laps keep the
    /// branches and the regions held back. One slot fills the region it contracts, so it has
    /// steady laps only where a region is held back. How many ran.
    fn steady_ahead_laps<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: &WalkExpand,
        stages: &mut StagesExpand<T>,
        total: NativeExpand<usize>,
        held_back: usize,
        fill: &mut Fill,
        compute: &mut F,
    ) -> NativeExpand<usize>
    where
        StagesExpand<T>: StagesFill,
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        let depth = stages.depth;
        let ahead = depth - 1 + held_back;
        if ahead == 0 {
            return 0usize.into_expand(scope);
        }
        let steady = steady_laps(scope, total, ahead, depth);
        let mut steady_body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let ahead = region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                let prefetch = walk.__expand_region_method(scope, ahead);
                fill(scope, stages, (j + depth - 1) % depth, &prefetch);
                // One slot: the fill is the region contracted next, published before it is read.
                if depth == 1 {
                    stages.publish(scope, FIRST_SLOT);
                }
                let region = walk.__expand_region_method(scope, region_idx);
                let slot = stages.placed_slot(scope, j, &region);
                compute(scope, slot, &region);
            }
        };
        run_laps(
            scope,
            0usize.into_expand(scope),
            steady,
            walk.unroll,
            &mut steady_body,
        );
        steady
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit of 128 takes two turns over 1024 values read 4 at a time.
    #[test]
    fn a_unit_takes_its_share_of_the_lines() {
        assert_eq!(Prefetch::turns(1024, 128, 4), 2);
    }

    /// A count of units the walk does not know deals no turns.
    #[test]
    #[should_panic(expected = "known count of units")]
    fn turns_need_a_known_count_of_units() {
        Prefetch::turns(1024, 0, 4);
    }

    /// A straight fill holds its loads whole; a decoding fill holds their words and a scale word
    /// per landed line.
    #[test]
    fn a_fill_holds_its_loads_in_registers() {
        assert_eq!(Prefetch::straight_registers(2, 8, 16), 8);
        assert_eq!(Prefetch::decoded_registers(2, 8, 4, 4), 2 * (1 + 2));
    }

    /// A prefetch is accepted up to its register budget, and refused past it.
    #[test]
    fn a_prefetch_fits_up_to_its_budget() {
        assert!(Prefetch::fits_registers(MOST_FETCHED_REGISTERS));
        assert!(!Prefetch::fits_registers(MOST_FETCHED_REGISTERS + 1));
    }
}
