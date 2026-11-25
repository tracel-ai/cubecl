use super::Move;
use crate::memory_management::MemoryLocation;
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
    /// The relocation of `moves`, whose copies landed.
    pub(super) fn new(moves: Vec<Move>) -> Self {
        Self { moves }
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
