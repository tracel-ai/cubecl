//! Who moves an operand's bytes into a stage, and whether they have landed when the fill
//! returns: the [`Delivery`].

use crate::{Refusal, Rendezvous, Space};

/// Who moves an operand into a stage, and whether its bytes have landed when the fill returns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    /// Every unit loads its own lines and stores them, through its registers: landed when the fill
    /// returns, and a line can be decoded on its way.
    #[default]
    SyncPerUnit,
    /// Every unit hands its own lines to the copy engine (`cp.async`), memory to memory: they land
    /// after the fill returns, as they lie, and something the slot waits on tracks them.
    AsyncPerUnit,
    /// One unit hands the whole stage to the copy engine as one contiguous run
    /// (`cp.async.bulk`): it lands after the fill returns, counted in bytes on the slot's barrier.
    AsyncBulk,
    /// One unit hands the stage's box to the TMA engine through a tensor map
    /// (`cp.async.bulk.tensor`): it lands after the fill returns, counted in bytes on the slot's
    /// barrier.
    Tma,
    /// Every unit computes its values from their coordinates: nothing is read, and the stage
    /// holds them when the fill returns.
    Procedural,
}

/// CUDA's cap on each TMA box dimension.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    /// The byte alignment a stage a [`Tma`](Delivery::Tma) box lands in starts on: the span a
    /// 128-byte swizzle repeats over, which the engine keys off the address, so a stage's first
    /// row takes the first key as the fragment reading it does. A launch sizing its shared memory
    /// counts what aligning a stage to it can cost.
    pub const TMA_STAGE_ALIGNMENT: usize = 1024;

    /// Whether this delivery moves `stage` in one transfer, and why not where it cannot.
    pub fn moves(self, stage: &Space) -> Result<(), Refusal> {
        match self {
            Delivery::SyncPerUnit
            | Delivery::AsyncPerUnit
            | Delivery::AsyncBulk
            | Delivery::Procedural => Ok(()),
            Delivery::Tma => match stage
                .extents()
                .into_iter()
                .find(|&(_, edge)| edge > TMA_MAX_BOX_DIM)
            {
                Some((axis, edge)) => Err(Refusal::BoxPastDescriptor {
                    axis,
                    edge,
                    most: TMA_MAX_BOX_DIM,
                }),
                None => Ok(()),
            },
        }
    }

    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// Whether the fill returns before its bytes land.
    pub(crate) fn is_async(&self) -> bool {
        match self {
            Delivery::SyncPerUnit | Delivery::Procedural => false,
            Delivery::AsyncPerUnit | Delivery::AsyncBulk | Delivery::Tma => true,
        }
    }

    /// The synchronization required to materialize this source in a staging slot: an mbarrier for
    /// every async delivery, counting a bulk copy's bytes or each unit's committed `cp.async`
    /// copies.
    pub(crate) fn rendezvous(&self) -> Rendezvous {
        if self.is_async() {
            Rendezvous::Barrier
        } else {
            Rendezvous::Cube
        }
    }

    /// Whether the copy writes through the async proxy and completes on the slot's barrier as
    /// transaction bytes (`cp.async.bulk`, TMA), so the barrier's initialization must be fenced
    /// into that proxy before the first copy. A per-unit `cp.async` stays in the generic proxy.
    pub(crate) fn through_async_proxy(&self) -> bool {
        matches!(self, Delivery::AsyncBulk | Delivery::Tma)
    }

    /// Whether every unit of the cube takes part in the fill, rather than one elected issuer.
    pub(crate) fn every_unit_fills(&self) -> bool {
        match self {
            Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::Procedural => true,
            Delivery::AsyncBulk | Delivery::Tma => false,
        }
    }
}
