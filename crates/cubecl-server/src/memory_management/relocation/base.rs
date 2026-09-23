//! Moving live allocations off outdated pages, so those pages go back to the
//! driver now rather than when their longest-lived allocation ends.
//!
//! A [`Relocation`] is planned by the pools it empties, copied on the device
//! through a [`CopyQueue`], and committed by those pools again from the
//! [`Landed`] relocation the copy step hands back: an allocation cannot move
//! before its bytes are where it moves to.

use super::CopyQueue;
use crate::memory_management::ManagedMemoryHandle;
use crate::server::ServerError;
use crate::storage::ComputeStorage;
use crate::storage::StorageHandle;
use alloc::vec::Vec;

/// Live allocations on outdated pages, each with a slice reserved on a current
/// page to receive it.
///
/// Nothing has moved yet: dropping it abandons the relocation with nothing
/// lost, the reserved slices freed with it.
#[derive(Debug)]
pub struct Relocation {
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

/// A [`Relocation`] whose copies have landed: what the pools commit, and only
/// [`Relocation::copy`] makes.
#[derive(Debug)]
pub struct Landed {
    moves: Vec<Move>,
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
    /// The allocation being moved, as its owners hold it.
    pub allocation: ManagedMemoryHandle,
    /// The slice reserved to receive it.
    pub target: ManagedMemoryHandle,
    /// The bytes to copy, or `None` when the source was carved under a dry run
    /// and never resolved: there is nothing behind it to copy.
    pub copy: Option<StorageCopy>,
}

impl Relocation {
    /// A relocation of `moves`, each with its target already reserved.
    pub(crate) fn new(moves: Vec<Move>) -> Self {
        Self { moves }
    }

    /// How many allocations move.
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Whether there is nothing to move.
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Copy every allocation's bytes on `queue`, waiting for the device
    /// before the first copy and for the copies after the last.
    ///
    /// The wait for the copies happens whether every one of them was enqueued
    /// or not, so a relocation abandoned here never frees a target a copy is
    /// still writing.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault a wait revealed. The
    /// relocation is abandoned either way.
    pub fn copy<Storage: ComputeStorage>(
        self,
        storage: &mut Storage,
        queue: &mut dyn CopyQueue<Storage>,
    ) -> Result<Landed, ServerError> {
        queue.wait_device()?;
        let enqueued = self
            .moves
            .iter()
            .filter_map(|relocated| relocated.copy.as_ref())
            .try_for_each(|copy| queue.copy(storage, copy));
        let landed = queue.wait_copies();
        enqueued?;
        landed?;
        Ok(Landed { moves: self.moves })
    }
}

impl Landed {
    /// The moves, for the pools that planned them to commit.
    pub(crate) fn into_moves(self) -> Vec<Move> {
        self.moves
    }
}
