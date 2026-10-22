//! [`Meeting`], the fill-vs-read rendezvous of one staging slot, and its [`Rendezvous`] strategy.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// How a slot rendezvouses its fill against its read, fixed at construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rendezvous {
    /// Cooperative copy, synchronized by one `sync_cube` per phase.
    Cube,
    /// Cooperative copy into a stage one plane owns, synchronized by one `sync_plane` per phase.
    ///
    /// Every plane of the cube still has to reach the same count of them: a backend with no
    /// plane-wide barrier (WGSL) lowers `sync_plane` to the workgroup's.
    Plane,
    /// Async copy (TMA, `cp.async`) over a `full`/`empty` mbarrier pair.
    Barrier,
}

impl Rendezvous {
    /// Join the rendezvous of a slot's sources over a walk `fillers` planes fill, for stages
    /// `owner` holds; `Barrier` wins, and a plane's own stages meet on `sync_plane`.
    ///
    /// Panics on a plane's own stages filled by a bulk copy or by planes set aside to fill: both
    /// meet on barriers the whole cube arms.
    pub(crate) fn for_deliveries(
        deliveries: &[Delivery],
        fillers: usize,
        owner: StageOwner,
    ) -> Rendezvous {
        assert!(
            !deliveries.is_empty(),
            "Slot: a slot must have at least one delivery"
        );
        if let StageOwner::Plane { .. } = owner {
            assert!(
                fillers == 0
                    && deliveries
                        .iter()
                        .all(|delivery| delivery.rendezvous() == Rendezvous::Cube),
                "Slot: a stage one plane owns is filled by that plane's units alone, so neither an \
                 async copy nor planes set aside to fill can deliver it"
            );
            return Rendezvous::Plane;
        }
        let floor = if fillers > 0 {
            Rendezvous::Barrier
        } else {
            Rendezvous::Cube
        };
        deliveries.iter().fold(floor, |sync, delivery| {
            match (sync, delivery.rendezvous()) {
                (Rendezvous::Barrier, _) | (_, Rendezvous::Barrier) => Rendezvous::Barrier,
                (Rendezvous::Cube, Rendezvous::Cube) => Rendezvous::Cube,
                (Rendezvous::Plane, _) | (_, Rendezvous::Plane) => {
                    unreachable!("a delivery asks for the cube or a barrier, never a plane")
                }
            }
        })
    }

    /// Whether a barrier slot needs every unit to publish its writes.
    pub(crate) fn collective_full(deliveries: &[Delivery]) -> bool {
        deliveries.iter().any(Delivery::every_unit_fills)
    }

    /// Whether a barrier slot's units copy asynchronously, which `full` must track before it flips.
    pub(crate) fn commits(deliveries: &[Delivery]) -> bool {
        deliveries.contains(&Delivery::AsyncPerUnit)
    }

    /// Whether a barrier slot's barriers must be fenced into the async proxy once initialized:
    /// only a bulk copy completes on them from there. The fence is sm90+, so a slot without one
    /// must not emit it.
    pub(crate) fn fences(deliveries: &[Delivery]) -> bool {
        deliveries.iter().any(Delivery::through_async_proxy)
    }
}

/// The rendezvous for one slot and the barriers it owns.
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum Meeting {
    /// Synchronous cooperative copy, synchronized by one `sync_cube` per phase.
    Cube,
    /// Synchronous cooperative copy into a stage one plane owns, synchronized by one `sync_plane`
    /// per phase ([`Rendezvous::Plane`]).
    Plane,
    /// Async producer/consumer over a `full`/`empty` mbarrier pair, one parity each.
    Barrier {
        /// Producer to consumer: flips once declared arrivals and TMA bytes land.
        full: Shared<Barrier>,
        /// Consumer to producer: flips once every reader has freed the slot.
        empty: Shared<Barrier>,
        /// Whether `full` counts every producer's arrival or only the elected issuer's.
        all_publish: bool,
        /// Whether each producer hands `full` its async copies before it arrives, so the phase
        /// waits for their bytes too ([`Delivery::AsyncPerUnit`]).
        commits: bool,
        /// The unit that issues this slot's bulk copies and declares their bytes.
        elected: u32,
        /// `full`'s parity, flipped by the producer's release.
        writes: u32,
        /// `empty`'s parity, flipped by the consumer's release.
        reads: u32,
    },
}

#[cube]
impl Meeting {
    /// Allocate the pipeline for `sync`; for [`Barrier`](Rendezvous::Barrier), arm and fence the
    /// mbarrier pair. Must run while every unit of the cube is present.
    pub(crate) fn new(
        #[comptime] sync: Rendezvous,
        #[comptime] collective_full: bool,
        #[comptime] commits: bool,
        #[comptime] fences: bool,
        #[comptime] fillers: usize,
    ) -> Meeting {
        match sync {
            Rendezvous::Cube => Meeting::new_Cube(),
            Rendezvous::Plane => Meeting::new_Plane(),
            Rendezvous::Barrier => {
                let full =
                    Barrier::shared(Meeting::producers(collective_full, fillers), UNIT_POS == 0);
                let empty = Barrier::shared(Meeting::consumers(fillers), UNIT_POS == 0);
                if comptime!(fences) {
                    sync_async_proxy_shared();
                }
                sync_cube();
                let elected = Meeting::elected(fillers);
                let all_publish = comptime!(collective_full || fillers > 0);
                Meeting::new_Barrier(full, empty, all_publish, commits, elected, 0, 0)
            }
        }
    }

    /// Units that arrive on `full`: the filling planes' where the cube sets planes aside, since
    /// they alone fill, whatever share of the fill is cooperative.
    pub(crate) fn producers(#[comptime] collective_full: bool, #[comptime] fillers: usize) -> u32 {
        if comptime!(fillers > 0) {
            comptime!(fillers as u32) * CUBE_DIM_X
        } else if comptime!(collective_full) {
            CUBE_DIM
        } else {
            1u32.runtime()
        }
    }

    /// Units that arrive on `empty`: the ones that read the slot.
    pub(crate) fn consumers(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            CUBE_DIM
        } else {
            CUBE_DIM - comptime!(fillers as u32) * CUBE_DIM_X
        }
    }

    /// The unit that issues a bulk copy and declares its bytes.
    pub(crate) fn elected(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            0u32.runtime()
        } else {
            Meeting::consumers(fillers)
        }
    }

    /// Fill staged `dst` from `src`, spread over the units `dst` names ([`FillUnits`]).
    pub(crate) fn fill(&self, dst: &mut Tile, src: &Tile) {
        // Bound before the match, which borrows the kind.
        let space = comptime!(dst.place.space.clone());
        let element = src.stage_element();
        if comptime!(element == StageElement::Decoded) {
            dst.dequantize_from(src);
        } else {
            self.fill_as_it_lies(dst, src, space);
            dst.fill_scales_from(src, self);
        }
    }

    /// [`fill`](Meeting::fill) a source that decodes nothing.
    fn fill_as_it_lies(&self, dst: &mut Tile, src: &Tile, #[comptime] space: Space) {
        let define!(Served) = dst.element;
        match self {
            Meeting::Barrier { full, elected, .. } => match (&mut dst.kind, &src.kind) {
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => {
                    // Single issuer: the transaction count is the elected unit's alone.
                    if UNIT_POS == *elected {
                        full.expect_tx(d.size_bytes::<Served>());
                        s.stage_into::<Served>(d, full);
                    }
                }
                (TileKind::Memory(d), TileKind::Memory(s)) => d.fill_from::<Served>(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => {
                    d.fill_procedural::<Served>(s, space)
                }
                _ => panic!("Meeting::fill: unsupported kind pairing"),
            },
            Meeting::Cube | Meeting::Plane => dst.copy_from(src),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_and_strided_share_a_cube_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(
                &[Delivery::Procedural, Delivery::SyncPerUnit],
                0,
                StageOwner::CUBE
            ),
            Rendezvous::Cube
        );
    }

    #[test]
    fn procedural_and_tma_share_a_barrier_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::Tma], 0, StageOwner::CUBE),
            Rendezvous::Barrier
        );
        assert!(Rendezvous::collective_full(&[
            Delivery::Procedural,
            Delivery::Tma
        ]));
    }

    /// An async copy lands after the fill returns, so only the barrier publishes it, and every
    /// unit issued a share it has to hand over.
    #[test]
    fn an_async_copy_rendezvouses_on_a_barrier_every_unit_commits_to() {
        let deliveries = [Delivery::AsyncPerUnit];
        assert_eq!(
            Rendezvous::for_deliveries(&deliveries, 0, StageOwner::CUBE),
            Rendezvous::Barrier
        );
        assert!(Rendezvous::collective_full(&deliveries));
        assert!(Rendezvous::commits(&deliveries));
        assert!(!Rendezvous::commits(&[
            Delivery::Tma,
            Delivery::SyncPerUnit
        ]));
    }

    /// A bulk copy has one issuer, as TMA does: its bytes are counted on `full`, not handed over
    /// by every unit.
    #[test]
    fn a_bulk_copy_keeps_its_single_producer_arrival() {
        let deliveries = [Delivery::AsyncBulk];
        assert_eq!(
            Rendezvous::for_deliveries(&deliveries, 0, StageOwner::CUBE),
            Rendezvous::Barrier
        );
        assert!(!Rendezvous::collective_full(&deliveries));
        assert!(!Rendezvous::commits(&deliveries));
    }

    /// Only a bulk copy completes on the barrier from the async proxy. `cp.async` stays in the
    /// generic proxy, and the fence it would otherwise emit does not exist before sm90.
    #[test]
    fn only_a_bulk_copy_fences_the_barrier_into_the_async_proxy() {
        assert!(Rendezvous::fences(&[Delivery::Tma]));
        assert!(Rendezvous::fences(&[
            Delivery::SyncPerUnit,
            Delivery::AsyncBulk
        ]));
        assert!(!Rendezvous::fences(&[Delivery::AsyncPerUnit]));
        assert!(!Rendezvous::fences(&[
            Delivery::SyncPerUnit,
            Delivery::Procedural
        ]));
    }

    #[test]
    fn pure_tma_keeps_its_single_producer_arrival() {
        assert!(!Rendezvous::collective_full(&[Delivery::Tma]));
    }

    #[test]
    fn a_filled_slot_rendezvouses_on_a_barrier_whatever_delivered_it() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::SyncPerUnit], 1, StageOwner::CUBE),
            Rendezvous::Barrier
        );
    }

    /// A plane's own stages meet on its own barrier, whatever copy fills them.
    #[test]
    fn a_planes_own_stage_rendezvouses_on_the_plane() {
        let owner = StageOwner::Plane { planes: 4 };
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::SyncPerUnit], 0, owner),
            Rendezvous::Plane
        );
    }

    /// A bulk copy's barrier is armed by the whole cube, which a plane walking its own stages
    /// never meets.
    #[test]
    #[should_panic(expected = "filled by that plane's units alone")]
    fn a_planes_own_stage_takes_no_bulk_copy() {
        Rendezvous::for_deliveries(&[Delivery::Tma], 0, StageOwner::Plane { planes: 4 });
    }

    /// Planes set aside to fill are planes the owning plane's units are not.
    #[test]
    #[should_panic(expected = "filled by that plane's units alone")]
    fn a_planes_own_stage_takes_no_filling_planes() {
        Rendezvous::for_deliveries(&[Delivery::SyncPerUnit], 1, StageOwner::Plane { planes: 4 });
    }
}
