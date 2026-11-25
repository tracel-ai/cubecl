//! Moving live allocations off outdated pages, so those pages go back to the
//! driver now rather than when their longest-lived allocation ends.
//!
//! A [`Relocation`] is planned by the pools it empties, copied on the device
//! through a [`RelocationCopier`], and committed by those pools again from the
//! [`LandedRelocation`] the copy step hands back: an allocation cannot move
//! before its bytes are where it moves to.

use super::{LandedRelocation, RelocationCopier};
use crate::memory_management::{ManagedMemoryBinding, ManagedMemoryHandle};
use crate::server::ServerError;
use crate::storage::StorageAllocator;
use crate::storage::StorageHandle;
use alloc::vec::Vec;

/// Live allocations on outdated pages, each with a slice reserved on a current
/// page to receive it.
///
/// # Guarantees
///
/// Nothing has moved yet: dropping it abandons the relocation with nothing
/// lost, the reserved slices freed with it.
#[derive(Debug)]
pub(crate) struct Relocation {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

/// Where a relocation may reserve the slices it moves allocations into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRoom {
    /// Only the room the current pages already have. Memory is what is short,
    /// and a relocation that allocated a page would spend more than it frees.
    Held,
    /// New pages of the current size too, when the room held runs out. Memory
    /// is not what is short, so emptying every outdated page is worth a page.
    MayAllocate,
}

/// One device copy a relocation needs, between two resolved storages of the
/// same size.
#[derive(Debug, Clone)]
pub struct StorageCopy {
    /// Where the bytes are now.
    pub source: StorageHandle,
    /// Where they go.
    pub target: StorageHandle,
}

/// One live allocation moving to a reserved slice: `allocation`'s handle is
/// handed over to the target once `copy` has landed.
#[derive(Debug)]
pub(crate) struct Move {
    /// The allocation being moved, bound rather than held, so the move does not count among
    /// its owners' handles while it runs.
    pub allocation: ManagedMemoryBinding,
    /// The slice reserved to receive it.
    pub target: ManagedMemoryHandle,
    /// The bytes to copy, or `None` when the source was carved under a process mode
    /// that discards launches and never resolved: there is nothing behind it to
    /// copy.
    pub copy: Option<StorageCopy>,
}

impl Relocation {
    /// Whether there is nothing to move.
    pub(crate) fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Copy every allocation's bytes with `copier`, waiting for the device
    /// before the first copy and for the copies after the last.
    ///
    /// # Guarantees
    ///
    /// The wait for the copies happens whether every one of them was enqueued
    /// or not, so a relocation abandoned here never frees a target a copy is
    /// still writing: when that wait fails on a device that is not poisoned,
    /// the targets and the allocations they were for stay reserved for good.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault a wait revealed. The
    /// relocation is abandoned either way.
    pub(crate) fn copy<Storage: StorageAllocator>(
        self,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
    ) -> Result<LandedRelocation, ServerError> {
        LandedRelocation::land(self.moves, storage, copier)
    }

    /// A relocation of `moves`, each with its target already reserved.
    pub(crate) fn new(moves: Vec<Move>) -> Self {
        Self { moves }
    }
}
