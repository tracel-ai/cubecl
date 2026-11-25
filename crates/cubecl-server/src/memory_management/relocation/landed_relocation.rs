use super::{Move, RelocationCopier};
use crate::memory_management::{MemoryLocation, release_or_leak};
use crate::server::ServerError;
use crate::storage::StorageAllocator;
use alloc::vec::Vec;

/// A [`Relocation`](super::Relocation) whose copies have landed: what the
/// pools commit, and only [`Relocation::copy`](super::Relocation::copy) makes.
#[derive(Debug)]
pub struct LandedRelocation {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

impl LandedRelocation {
    /// How many allocations moved.
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Whether nothing moved.
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

impl LandedRelocation {
    /// Copy the bytes of every one of `moves` with `copier`, waiting for the
    /// device before the first copy and for the copies after the last, and
    /// hand the moves back landed once every copy did. The wait for the
    /// copies happens whether every one of them was enqueued or not, and when
    /// it fails on a device that is not poisoned, the moves are leaked rather
    /// than freed under a copy still writing.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault a wait revealed.
    pub(super) fn land<Storage: StorageAllocator>(
        moves: Vec<Move>,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
    ) -> Result<Self, ServerError> {
        copier.wait_device()?;
        let enqueued = moves
            .iter()
            .filter_map(|relocated| relocated.copy.as_ref())
            .try_for_each(|copy| copier.copy(storage, copy));
        if let Err(err) = copier.wait_copies() {
            release_or_leak(moves, &err);
            enqueued?;
            return Err(err);
        }
        enqueued?;
        Ok(Self { moves })
    }
}

/// The location each allocation of `landed` moves from and the one it moves
/// to, read from the handles, which the pools keep current: a move stores no
/// location of its own, so it can never hold a stale one. Each move's handles
/// are dropped before its locations are yielded, so only the slices hold them
/// once the pools hand the allocation over.
pub(crate) fn landed_locations(
    landed: LandedRelocation,
) -> impl Iterator<Item = (MemoryLocation, MemoryLocation)> {
    landed.moves.into_iter().map(|relocated| {
        let source = relocated.allocation.descriptor().location();
        let destination = relocated.target.descriptor().location();
        (source, destination)
    })
}
