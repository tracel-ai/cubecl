use super::Move;
use alloc::vec::Vec;

/// The moves the pools commit, which only
/// [`RelocationLander::land`](super::RelocationLander::land) hands back.
#[derive(Debug)]
pub struct LandedRelocation {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

impl LandedRelocation {
    /// The relocation of `moves`, whose copies every later use of their
    /// storages runs after.
    pub fn new(moves: Vec<Move>) -> Self {
        Self { moves }
    }

    /// The moves, for the pools that planned them to commit.
    pub fn into_moves(self) -> alloc::vec::IntoIter<Move> {
        self.moves.into_iter()
    }
}
