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
            |scope, stages, slot, region| stages.fill_streamed(scope, slot, region),
            compute,
        )
    }
}

/// Where a walk's next stage waits while the previous one is contracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prefetch {
    /// In its own slot, filled before the previous contraction.
    InSlots,
    /// In registers, loaded before the contraction and stored after; bounded by
    /// [`fits`](Prefetch::fits).
    InRegisters,
}

impl Prefetch {
    /// Scalars one unit holds when `units` prefetch `elements` values read `width` at a time.
    pub fn scalars(elements: usize, units: usize, width: usize) -> usize {
        UnitLines::new(elements.div_ceil(width), units).scalars(width)
    }

    /// Whether a register prefetch of `scalars` a unit, over both operands, is accepted.
    pub fn fits(scalars: usize) -> bool {
        scalars <= MOST_FETCHED_SCALARS
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
            Prefetch::InSlots => self.__expand_pipelined_method(scope, walk, compute),
            Prefetch::InRegisters => self.__expand_prefetched_method(scope, walk, compute),
        }
    }
}

impl Stages<OperandPair> {
    /// [`pipelined`](Stages::pipelined) with the next region's loads held in registers across the
    /// current contraction.
    ///
    /// Requires depth 1 or 2, every unit copying straight from a plain operand, and loads within
    /// [`Prefetch::fits`]; otherwise refused at expansion. Fixed operands, and operands whose
    /// stages keep their scales beside them, fall back to [`pipelined`](Stages::pipelined).
    pub fn prefetched<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<OperandPair>, &Region),
    {
        unexpanded!()
    }
}

impl StagesExpand<OperandPair> {
    pub fn __expand_prefetched_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair>, &RegionExpand),
    {
        // Registers hold the values alone, so scales a stage keeps beside them are brought by the
        // slots.
        let scales_beside = self.sources.lhs.__expand_scales_beside_method(scope)
            || self.sources.rhs.__expand_scales_beside_method(scope);
        if self.has_fixed(scope) || scales_beside {
            return self.__expand_pipelined_method(scope, walk, compute);
        }
        self.__expand_assert_copied_by_every_unit_method(scope);
        assert!(
            self.depth <= 2,
            "Stages::prefetched: {} slots hold one region ahead, so the slots past a second \
             would sit idle; build them with a depth of 1 or 2",
            self.depth
        );
        let total = walk.__expand_total_method(scope);
        self.prime(scope, &walk, &total);
        self.prefetching_laps(scope, walk, total, compute);
    }

    /// Fill and publish the first region before any contraction.
    fn prime(&mut self, scope: &Scope, walk: &WalkExpand, total: &NativeExpand<usize>) {
        let any = 0usize.into_expand(scope).__expand_lt_method(scope, total);
        if_expand(scope, any, |scope| {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            self.fill_streamed(scope, FIRST_SLOT, &first);
            self.publish(scope, FIRST_SLOT);
        });
    }

    /// The laps whose every region has a next one to fetch, run with no branch around the fetch,
    /// so what it computes that no lap changes (each unit's place in the stage) leaves the loop.
    /// How many ran.
    fn steady_prefetching_laps<Lhs: Numeric, Rhs: Numeric, F>(
        &mut self,
        scope: &Scope,
        walk: &WalkExpand,
        total: NativeExpand<usize>,
        lhs: &mut <Array<Vector<Lhs, super::base::FL>> as CubeType>::ExpandType,
        rhs: &mut <Array<Vector<Rhs, super::base::FR>> as CubeType>::ExpandType,
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
                self.__expand_fetch_method(scope, target, &sources, &upcoming, lhs, rhs);
                let region = walk.__expand_region_method(scope, region_idx);
                let slot = self.__expand_slot_mut_method(scope, j);
                compute(scope, slot, &region);
                // One slot: every unit has read it before any overwrites it.
                if depth == 1 {
                    self.publish(scope, FIRST_SLOT);
                }
                self.__expand_store_method(scope, target, lhs, rhs);
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
        let depth = self.depth;
        // The registers a fetch holds a region in are at the elements the sources hold.
        define_scalar!(Lhs);
        define_scalar!(Rhs);
        let _lhs = scope.define_type::<Lhs>(self.sources.lhs.element);
        let _rhs = scope.define_type::<Rhs>(self.sources.rhs.element);
        let (mut lhs, mut rhs) = self.__expand_fetch_buffers_method::<Lhs, Rhs>(scope);
        let steady =
            self.steady_prefetching_laps(scope, &walk, total, &mut lhs, &mut rhs, &mut compute);
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
                    if_expand(scope, prefetching, |scope| {
                        let upcoming = walk.__expand_region_method(scope, next);
                        let sources = self.sources.clone();
                        self.__expand_fetch_method(
                            scope, target, &sources, &upcoming, &mut lhs, &mut rhs,
                        );
                    });
                    let region = walk.__expand_region_method(scope, region_idx);
                    let slot = self.__expand_slot_mut_method(scope, j);
                    compute(scope, slot, &region);
                    if_expand(scope, prefetching, |scope| {
                        // One slot: every unit must finish reading before any overwrites it.
                        if depth == 1 {
                            self.publish(scope, FIRST_SLOT);
                        }
                        self.__expand_store_method(scope, target, &lhs, &rhs);
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
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand),
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
                fill(scope, stages, slot, &region);
            });
        }

        let laps = total
            .__expand_plus_method(scope, (depth - 1).into_expand(scope))
            .__expand_divided_by_method(scope, depth.into_expand(scope));
        // Over a run ([`Walk::run`]) the steady laps leave its last region to the last laps and fill
        // unchecked.
        let steady = match walk.run {
            None => steady_ahead_laps(scope, &walk, stages, total, 0, &mut fill, &mut compute),
            Some(_) => steady_ahead_laps(
                scope,
                &walk,
                stages,
                total,
                1,
                &mut |scope, stages: &mut StagesExpand<T>, slot, region: &RegionExpand| {
                    stages.fill_inside_run(scope, slot, region)
                },
                &mut compute,
            ),
        };
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                if depth == 1 {
                    let region = walk.__expand_region_method(scope, region_idx);
                    fill(scope, stages, FIRST_SLOT, &region);
                    stages.publish(scope, FIRST_SLOT);
                    let slot = stages.__expand_slot_mut_method(scope, FIRST_SLOT);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let in_walk = region_idx.__expand_lt_method(scope, &total);
                    // Compute is emitted once, after the fill branch, keeping its accumulator live.
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        fill(scope, stages, (j + depth - 1) % depth, &prefetch);
                    })
                    .or_else(scope, |scope| {
                        // Draining: no fill follows, so this consume publishes.
                        if_expand(scope, in_walk, |scope| {
                            stages.publish(scope, j);
                        });
                    });
                    if_expand(scope, in_walk, |scope| {
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = stages.__expand_slot_mut_method(scope, j);
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
                let slot = stages.__expand_slot_mut_method(scope, j);
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
