use super::Move;
use alloc::vec::Vec;

/// A relocation whose copies have landed: what the pools commit, and only
/// [`RelocationLander::land`](super::RelocationLander::land) makes.
#[derive(Debug)]
pub(crate) struct LandedRelocation {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

impl LandedRelocation {
    /// The relocation of `moves`, whose copies landed.
    pub(super) fn new(moves: Vec<Move>) -> Self {
        Self { moves }
    }

    /// The moves, for the pools that planned them to commit.
    pub(crate) fn into_moves(self) -> Vec<Move> {
        self.moves
    }
}
