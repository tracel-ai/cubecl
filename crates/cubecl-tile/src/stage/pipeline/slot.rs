//! [`Slot`]: a payload plus the [`Meeting`] sequencing its fill against its read.

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::payload::base::{Payload, PayloadExpand};
use super::payload::pair::OperandPair;
use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// One slot of a buffered walk: payload `T` and the rendezvous sequencing fill against read.
#[derive(CubeType)]
pub struct Slot<T: CubeType> {
    pub(crate) data: T,
    pub(crate) meeting: Meeting,
    /// When each operand of `T` is filled, in the order `T` holds them.
    #[cube(comptime)]
    pub(crate) refills: Vec<Refill>,
}

#[cube]
impl<T: CubeType> Slot<T> {
    /// Wrap an already-built payload and meeting.
    pub(crate) fn wrap(data: T, meeting: Meeting, #[comptime] refills: Vec<Refill>) -> Slot<T> {
        Slot::<T> {
            data,
            meeting,
            refills,
        }
    }

    /// Whether this slot has any fixed operand.
    #[allow(dead_code)] // Reached through its expand, from [`StagesFill`].
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(self.refills.contains(&Refill::Once))
    }

    /// Publish this slot's last fill where no later fill will; call right before
    /// [`consume`](Slot::consume).
    pub fn publish(&self) {
        self.meeting.publish();
    }
}

#[cube]
impl<P: Payload<P> + CubeType<ExpandType: Clone>> Slot<P> {
    /// Bring `src`'s window at `region` into this slot, for every operand whose refill is `only`.
    pub(crate) fn refill(&mut self, src: &P, region: &Region, #[comptime] only: Refill) {
        let refills = comptime!(self.refills.clone());
        let writes = comptime!(refills.contains(&only));
        if comptime!(writes || only == Refill::EveryRegion) {
            self.meeting.acquire_write();
            self.data
                .bring(src, &self.meeting, region, comptime!(refills), only);
            self.meeting.release_write();
        }
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
        self.meeting.__expand_acquire_read_method(scope);
        compute(scope, &self.data.lhs, &self.data.rhs);
        self.meeting.__expand_release_read_method(scope);
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
        self.meeting.__expand_acquire_read_method(scope);
        compute(scope, &self.data);
        self.meeting.__expand_release_read_method(scope);
    }
}
