//! The opaque dimension label every layout and space concept is keyed on.

use cubecl::zspace::SmallVec;
use serde::{Deserialize, Serialize};

use crate::Space;

/// A labeled dimension. The `u8` is a client-assigned index, not a position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Dimension(pub u8);

/// A comptime map from [`Dimension`] to a value, in declared (canonical) order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct DimensionMap<T: Copy> {
    entries: SmallVec<[(Dimension, T); Space::MAX_RANK]>,
}

impl<T: Copy> DimensionMap<T> {
    /// Order is significant.
    pub(crate) fn new(entries: &[(Dimension, T)]) -> Self {
        DimensionMap {
            entries: SmallVec::from_slice(entries),
        }
    }

    pub(crate) fn get(&self, dimension: Dimension) -> T {
        self.entries
            .iter()
            .find(|(a, _)| *a == dimension)
            .expect("DimensionMap::get: dimension not present")
            .1
    }

    pub(crate) fn dimension_at(&self, i: usize) -> Dimension {
        self.entries[i].0
    }

    /// The dimensions in declared order.
    pub(crate) fn dimensions(&self) -> impl Iterator<Item = Dimension> + '_ {
        self.entries.iter().map(|&(a, _)| a)
    }

    pub(crate) fn position(&self, dimension: Dimension) -> usize {
        self.entries
            .iter()
            .position(|(a, _)| *a == dimension)
            .expect("DimensionMap::position: dimension not present")
    }

    pub(crate) fn contains(&self, dimension: Dimension) -> bool {
        self.entries.iter().any(|(a, _)| *a == dimension)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
