//! [`Count`]: how many tiles a level takes along one of its axes.

use crate::Extent;

/// How many tiles a level takes along one of its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Count {
    /// This many, stated: a walk's steps, or a scope's workers with one tile each.
    Stated(usize),
    /// Every tile the level above hands down; the one count whose tile can overhang.
    All,
    /// Every tile, distributed across this many workers in runs.
    AllAcross(usize),
    /// This many tiles, taken in turns by however many units the launch runs.
    Distributed(usize),
}

/// What a walk counts along one axis of a level: a constant, or the extent handed down.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GridCount {
    Const(usize),
    Extent(usize),
}

impl Count {
    /// The workers the tiles are distributed to, where that is a stated number.
    pub(crate) fn stated(self) -> Option<usize> {
        match self {
            Count::Stated(n) | Count::AllAcross(n) => Some(n),
            Count::All | Count::Distributed(_) => None,
        }
    }

    /// Tiles this count takes along an axis of `extent`; a partial last tile counts.
    pub(crate) fn tiles(self, extent: usize, tile: usize) -> usize {
        match self {
            Count::Stated(n) | Count::Distributed(n) => n,
            Count::All | Count::AllAcross(_) => extent.div_ceil(tile),
        }
    }

    /// [`tiles`](Count::tiles) where the host can prove it.
    pub(crate) fn tiles_const(self, extent: Extent, tile: usize) -> Option<usize> {
        match (self, extent) {
            (Count::Stated(n) | Count::Distributed(n), _) => Some(n),
            (_, Extent::Static(extent)) => Some(self.tiles(extent, tile)),
            (_, Extent::Dynamic) => None,
        }
    }

    /// What a walk counts along an axis stepped in `tile`.
    pub(crate) fn grid(self, tile: usize) -> GridCount {
        match self {
            Count::Stated(n) | Count::Distributed(n) => GridCount::Const(n),
            Count::All | Count::AllAcross(_) => GridCount::Extent(tile),
        }
    }
}

/// A count a units level states along an axis, as the units it takes.
impl From<Count> for usize {
    fn from(count: Count) -> usize {
        count
            .stated()
            .expect("a units level's count is stated; every tile is the launch's")
    }
}
