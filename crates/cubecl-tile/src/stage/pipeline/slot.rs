//! [`Slot`]: a payload plus the [`Meeting`] sequencing its fill against its read.

use cubecl::prelude::*;

use super::payload::base::{Payload, PayloadExpand};
use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// One slot of a buffered walk: payload `T` and the rendezvous sequencing fill against read.
#[derive(CubeType)]
pub struct Slot<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Meeting,
    /// When each operand of `T` is filled, in the order `T` holds them.
    #[cube(comptime)]
    pub(crate) refills: Vec<Refill>,
}

#[cube]
impl<T: CubeType> Slot<T> {
    /// Wrap an already-built payload and pipeline.
    pub(crate) fn wrap(data: T, pipeline: Meeting, #[comptime] refills: Vec<Refill>) -> Slot<T> {
        Slot::<T> {
            data,
            pipeline,
            refills,
        }
    }

    /// Whether this slot has any fixed operand.
    #[allow(dead_code)] // Reached through its expand, from [`StagesFill`].
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(self.refills.contains(&Refill::Once))
    }

    /// Producer acquire: wait until the slot is free; a `Plane` slot waits on `sync_plane`.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_write(&self) {
        match &self.pipeline {
            Meeting::Barrier { empty, writes, .. } => empty.wait_parity(*writes ^ 1),
            Meeting::Cube => sync_cube(),
            Meeting::Plane => sync_plane(),
        }
    }

    /// Producer release: publish the fill, async copies included ([`Meeting::producers`] say
    /// who arrives).
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn release_write(&mut self) {
        match &mut self.pipeline {
            Meeting::Barrier {
                full,
                all_publish,
                commits,
                elected,
                writes,
                ..
            } => {
                if *all_publish || UNIT_POS == *elected {
                    // Handed over before the arrival, so the phase cannot flip on the arrivals
                    // with this unit's copies still in flight.
                    if *commits {
                        full.commit_copy_async();
                    }
                    full.arrive();
                }
                *writes ^= 1;
            }
            Meeting::Cube | Meeting::Plane => {}
        }
    }

    /// Consumer acquire: wait for the slot's fill.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_read(&self) {
        match &self.pipeline {
            Meeting::Barrier { full, reads, .. } => full.wait_parity(*reads),
            Meeting::Cube | Meeting::Plane => {}
        }
    }

    /// Consumer release: free the slot.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn release_read(&mut self) {
        match &mut self.pipeline {
            Meeting::Barrier { empty, reads, .. } => {
                empty.arrive();
                *reads ^= 1;
            }
            Meeting::Cube | Meeting::Plane => {}
        }
    }

    /// Publish this slot's last fill where no later fill will; call right before
    /// [`consume`](Slot::consume).
    pub fn publish(&self) {
        match &self.pipeline {
            Meeting::Cube => sync_cube(),
            Meeting::Plane => sync_plane(),
            Meeting::Barrier { .. } => {}
        }
    }
}

#[cube]
impl<P: Payload<P> + CubeType<ExpandType: Clone>> Slot<P> {
    /// Bring `src`'s window at `region` into this slot, for every operand whose refill is `only`.
    pub(crate) fn refill(&mut self, src: &P, region: &Region, #[comptime] only: Refill) {
        let refills = comptime!(self.refills.clone());
        let writes = comptime!(refills.contains(&only));
        if comptime!(writes || only == Refill::EveryRegion) {
            self.acquire_write();
            self.data
                .bring(src, &self.pipeline, region, comptime!(refills), only);
            self.release_write();
        }
    }
}
