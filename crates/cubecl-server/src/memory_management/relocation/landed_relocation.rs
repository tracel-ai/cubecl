use super::Move;
use alloc::vec::Vec;

/// A [`Relocation`](super::Relocation) whose copies have landed: what the pools commit, and only
/// [`Relocation::copy`](super::Relocation::copy) makes.
#[derive(Debug)]
pub struct LandedRelocation {
    pub(super) moves: Vec<Move>,
}

impl LandedRelocation {
    /// The moves, for the pools that planned them to commit.
    pub(crate) fn into_moves(self) -> Vec<Move> {
        self.moves
    }
}
