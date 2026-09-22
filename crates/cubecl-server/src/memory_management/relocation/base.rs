//! Moving live allocations off outdated pages, so those pages go back to the
//! driver now rather than when their longest-lived allocation ends.
//!
//! A [`Relocation`] is planned by
//! [`MemoryManagement::relocation`](crate::memory_management::MemoryManagement::relocation),
//! copied on the device through a [`CopyQueue`], and committed by
//! [`MemoryManagement::commit_relocation`](crate::memory_management::MemoryManagement::commit_relocation),
//! which only takes the [`Landed`] relocation the copy step hands back: an
//! allocation cannot move before its bytes are where it moves to.

use crate::memory_management::ManagedMemoryHandle;
use crate::memory_management::drop_queue::Fence;
use crate::server::{IoError, ServerError};
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

/// A [`Relocation`] whose copies have landed: what
/// [`MemoryManagement::commit_relocation`](crate::memory_management::MemoryManagement::commit_relocation)
/// takes, and only [`Relocation::copy`] makes.
#[derive(Debug)]
pub struct Landed {
    moves: Vec<Move>,
}

/// Where a relocation's bytes are copied: a device queue, and the fence that
/// says the copies enqueued on it so far have landed.
pub trait CopyQueue {
    /// What [`fence`](Self::fence) hands back.
    type Fence: Fence;

    /// Enqueue copying `copy.source`'s bytes into `copy.target`.
    ///
    /// # Errors
    ///
    /// The device's refusal to copy.
    fn copy(&mut self, copy: &StorageCopy) -> Result<(), IoError>;

    /// A fence past every copy enqueued so far.
    fn fence(&mut self) -> Self::Fence;
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

    /// Copy every allocation's bytes on `queue` and wait until they have
    /// landed.
    ///
    /// The wait happens whether every copy was enqueued or not, so a
    /// relocation abandoned here never frees a target a copy is still writing.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault the wait revealed. The
    /// relocation is abandoned either way.
    pub fn copy(self, queue: &mut impl CopyQueue) -> Result<Landed, ServerError> {
        let enqueued = self
            .moves
            .iter()
            .filter_map(|relocated| relocated.copy.as_ref())
            .try_for_each(|copy| queue.copy(copy));
        let landed = queue.fence().wait();
        enqueued?;
        landed?;
        Ok(Landed { moves: self.moves })
    }
}

impl Landed {
    /// The moves, for the pools that hold their allocations to hand over.
    pub(crate) fn into_moves(self) -> Vec<Move> {
        self.moves
    }
}
