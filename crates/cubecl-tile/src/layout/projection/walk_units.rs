//! The step of a walk over a projection, as runtime counts of compile-time units ([`WalkUnits`]).

use cubecl::zspace::SmallVec;

use crate::{Dimension, Space};

/// One dimension a walk moves along, and how far one count of it moves.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct WalkedDimension {
    pub dimension: Dimension,
    /// The coordinates of `dimension` one count moves, in the units its coordinate counts: lines
    /// along the innermost dimension, values along the others.
    pub coords_per_count: usize,
}

/// The dimensions a walk moves along, each with the compile-time unit its runtime count counts.
///
/// # Assumptions
///
/// - Each dimension appears once.
/// - Every `coords_per_count` is past `0`.
///
/// # Guarantees
///
/// - A walk's counts are read in the order of `dimensions`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct WalkUnits {
    pub dimensions: SmallVec<[WalkedDimension; Space::MAX_RANK]>,
}

impl WalkUnits {
    /// The walk that moves nothing: an anchor read where it was taken.
    pub(crate) fn none() -> Self {
        WalkUnits {
            dimensions: SmallVec::new(),
        }
    }

    /// The walk along `dimension` alone.
    pub(crate) fn along(dimension: Dimension, coords_per_count: usize) -> Self {
        WalkUnits::over(&[(dimension, coords_per_count)])
    }

    /// The walk along each of `dimensions`, its counts in that order.
    pub(crate) fn over(dimensions: &[(Dimension, usize)]) -> Self {
        for (at, &(dimension, coords_per_count)) in dimensions.iter().enumerate() {
            assert!(
                coords_per_count > 0,
                "WalkUnits: a count along {dimension:?} moves no coordinate; leave it out of the walk"
            );
            assert!(
                dimensions[..at]
                    .iter()
                    .all(|&(earlier, _)| earlier != dimension),
                "WalkUnits: {dimension:?} is walked twice; state one unit a dimension"
            );
        }
        WalkUnits {
            dimensions: dimensions
                .iter()
                .map(|&(dimension, coords_per_count)| WalkedDimension {
                    dimension,
                    coords_per_count,
                })
                .collect(),
        }
    }

    /// The coordinates of `dimension` one count of this walk moves, `None` where it does not
    /// move it.
    pub(crate) fn coords_per_count(&self, dimension: Dimension) -> Option<usize> {
        self.position(dimension)
            .map(|at| self.dimensions[at].coords_per_count)
    }

    /// Where `dimension`'s count sits among this walk's counts, `None` where it does not move it.
    pub(crate) fn position(&self, dimension: Dimension) -> Option<usize> {
        self.dimensions
            .iter()
            .position(|walked| walked.dimension == dimension)
    }
}

/// A walk read in the kernel's expansion, where it is only ever known at compile time.
mod walk_expand {
    use super::WalkUnits;
    use crate::comptime_only;
    use cubecl::prelude::*;

    comptime_only!(WalkUnits);
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Dimension = Dimension(0);
    const B: Dimension = Dimension(1);

    #[test]
    fn a_walk_answers_its_units_in_the_order_stated() {
        let walk = WalkUnits::over(&[(B, 4), (A, 1)]);
        assert_eq!(walk.position(B), Some(0));
        assert_eq!(walk.coords_per_count(A), Some(1));
        assert_eq!(walk.coords_per_count(Dimension(2)), None);
    }

    #[test]
    #[should_panic(expected = "is walked twice")]
    fn a_walk_refuses_a_repeated_dimension() {
        WalkUnits::over(&[(A, 1), (B, 2), (A, 4)]);
    }

    #[test]
    #[should_panic(expected = "moves no coordinate")]
    fn a_walk_refuses_a_zero_unit() {
        WalkUnits::along(A, 0);
    }
}
