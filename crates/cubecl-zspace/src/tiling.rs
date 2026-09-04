//! Storage tiling: how many fragments each logical dim is stored as.
//!
//! A `u16`, two bits per logical dim, dim `i` at bits `2i..2i + 2`, holding
//! its number of fragments minus one. `0` is untiled.
//!
//! | Limit | |
//! |---|---|
//! | Logical dims | 8 |
//! | Fragments of one logical dim | 4, so 3 levels of tiling |
//!
//! The physical dims are level-major, coarsest first: every logical dim's
//! coarsest fragment in logical order, then every dim still deep enough gives
//! its next, down to the finest. `[k, n]` stored `[k/32, n/32, 32, 32]` is
//! `[2, 2]`, and a logical extent is its fragments' product.

use alloc::format;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::MetadataError;

/// The most logical dims a tiling describes: two bits each in a `u16`.
pub const MAX_LOGICAL_DIMS: usize = 8;

/// The most fragments one logical dim is stored as.
pub const MAX_FRAGMENTS: usize = 4;

/// How many fragments each logical dim is stored as. See the [module doc](self).
///
/// `Default` is untiled, and so is `0`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Tiling(u16);

impl Tiling {
    /// Every logical dim is one fragment.
    pub const UNTILED: Tiling = Tiling(0);

    /// The tiling whose logical dim `i` is stored as `fragments[i]` fragments.
    ///
    /// # Errors
    ///
    /// More than [`MAX_LOGICAL_DIMS`] dims, or a count outside `1..=MAX_FRAGMENTS`.
    pub fn new(fragments: &[usize]) -> Result<Tiling, MetadataError> {
        let rule = |holds: bool, name: &str| {
            holds
                .then_some(())
                .ok_or_else(|| invalid(format!("tiling {fragments:?}: {name}")))
        };
        rule(
            fragments.len() <= MAX_LOGICAL_DIMS,
            "at most 8 logical dims",
        )?;
        rule(
            fragments.iter().all(|f| (1..=MAX_FRAGMENTS).contains(f)),
            "1 to 4 fragments a dim",
        )?;

        let field = |(dim, &count): (usize, &usize)| ((count - 1) as u16) << (2 * dim);
        Ok(Tiling(fragments.iter().enumerate().map(field).sum()))
    }

    /// The number of fragments of each of `logical_rank` logical dims.
    pub fn fragments(self, logical_rank: usize) -> SmallVec<[usize; MAX_LOGICAL_DIMS]> {
        (0..logical_rank).map(|dim| self.levels(dim) + 1).collect()
    }

    /// The logical rank of a buffer of `rank` physical dims: one per dim, less
    /// the extra fragments.
    ///
    /// # Errors
    ///
    /// When `rank` holds fewer dims than the tiling's fragments, or a dim past
    /// the logical rank is tiled.
    pub fn logical_rank(self, rank: usize) -> Result<usize, MetadataError> {
        let extra: usize = (0..MAX_LOGICAL_DIMS).map(|dim| self.levels(dim)).sum();
        let logical = rank
            .checked_sub(extra)
            .filter(|&logical| Tiling::new(&self.fragments(logical)) == Ok(self))
            .ok_or_else(|| invalid(format!("{self:?} does not fit a rank-{rank} buffer")))?;
        Ok(logical)
    }

    /// Whether any logical dim is stored as more than one fragment.
    pub fn is_tiled(self) -> bool {
        self.0 != 0
    }

    fn levels(self, dim: usize) -> usize {
        ((self.0 >> (2 * dim)) & 0b11) as usize
    }
}

fn invalid(reason: alloc::string::String) -> MetadataError {
    MetadataError::Invalid { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragments_round_trip() {
        let fragments = [1, 2, 2];
        let tiling = Tiling::new(&fragments).unwrap();
        assert!(tiling.is_tiled());
        assert_eq!(tiling.fragments(3).as_slice(), &fragments);
        assert_eq!(Tiling::new(&[2, 2]).unwrap(), Tiling(0b01_01));
        assert_eq!(Tiling::new(&[1, 1, 1]).unwrap(), Tiling::UNTILED);
        assert_eq!(Tiling::UNTILED.fragments(2).as_slice(), &[1, 1]);
    }

    #[test]
    fn logical_rank_is_the_physical_rank_less_the_extra_fragments() {
        let tiling = Tiling::new(&[1, 2, 2]).unwrap();
        assert_eq!(tiling.logical_rank(5), Ok(3));
        assert!(tiling.logical_rank(4).is_err());
        assert!(tiling.logical_rank(2).is_err());
        assert_eq!(Tiling::UNTILED.logical_rank(4), Ok(4));
    }

    #[test]
    fn refuses_what_the_protocol_cannot_hold() {
        assert!(Tiling::new(&[1; 9]).is_err());
        assert!(Tiling::new(&[5]).is_err());
        assert!(Tiling::new(&[0]).is_err());
    }
}
