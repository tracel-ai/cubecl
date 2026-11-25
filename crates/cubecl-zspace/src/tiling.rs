//! Storage tiling: the pieces each logical dim is stored as, and the physical dim each piece is.
//!
//! A `u64`. Its low half says how many pieces: four bits per logical dim, dim `i` at bits
//! `4i..4i + 4`, holding its number of fragments minus one. Its high half says in which order the
//! physical dims list them: the place of that order among every order of as many dims, the sorted
//! one first. `0` is untiled.
//!
//! | Limit | |
//! |---|---|
//! | Logical dims | 8 |
//! | Fragments of one logical dim | 16, so 15 levels of tiling |
//! | Physical dims in another order than level-major | 12, whose orders a `u32` counts |
//!
//! Sixteen, because a level is whatever a reader has to find whole: a packed
//! word, a vector read, a storage tile, and the grid over them already take
//! four pieces of one dim.
//!
//! A logical dim's fragments are its pieces, coarsest first, and its extent is their product,
//! whatever physical dims they are. Level-major is the order `0` stands for: every logical dim's
//! coarsest fragment in logical order, then every dim still deep enough gives its next, down to
//! the finest. `[k, n]` stored `[k/32, n/32, 32, 32]` is `[2, 2]` level-major. A layout made for a
//! tensor core lists them otherwise: `[m, k]` stored `[m/16, k/16, m%8, k/2%4, k/8%2, m/8%2, k%2]`
//! holds `m`'s pieces in dims `[0, 5, 2]` and `k`'s in `[1, 4, 3, 6]`, whose finest are not the
//! last.

use alloc::format;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::MetadataError;

/// The most logical dims a tiling describes: four bits each in a `u32`.
pub const MAX_LOGICAL_DIMS: usize = 8;

/// The most fragments one logical dim is stored as.
pub const MAX_FRAGMENTS: usize = 16;

/// The most physical dims a tiling lists in another order than level-major: twelve have fewer
/// orders than a `u32` counts, and thirteen have more.
pub const MAX_ORDERED_DIMS: usize = 12;

/// Where the order's bits start, past the four a logical dim takes.
const ORDER_SHIFT: usize = 4 * MAX_LOGICAL_DIMS;

/// A value for each physical dim of a tiled buffer.
type PerPhysicalDim = SmallVec<[usize; MAX_ORDERED_DIMS]>;

/// The physical dims each logical dim is stored as, coarsest first.
type PhysicalDims = SmallVec<[SmallVec<[usize; MAX_FRAGMENTS]>; MAX_LOGICAL_DIMS]>;

/// The fragments each logical dim is stored as, and the physical dim each one is.
///
/// `Default` is untiled, and so is `0`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Tiling(u64);

impl Tiling {
    /// Every logical dim is one fragment.
    pub const UNTILED: Tiling = Tiling(0);

    /// The level-major tiling whose logical dim `i` is stored as `fragments[i]` fragments.
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
            "1 to 16 fragments a dim",
        )?;

        let field = |(dim, &count): (usize, &usize)| ((count - 1) as u64) << (4 * dim);
        Ok(Tiling(fragments.iter().enumerate().map(field).sum()))
    }

    /// The tiling whose logical dim `i` is stored as the physical dims `dims[i]`, coarsest first:
    /// what [`physical_dims`](Self::physical_dims) reads back.
    ///
    /// # Errors
    ///
    /// More than [`MAX_LOGICAL_DIMS`] dims, one stored as a number of fragments outside
    /// `1..=MAX_FRAGMENTS`, physical dims that are not `0..` each once, untiled dims listed in
    /// another order than their own, which strides say, or more than [`MAX_ORDERED_DIMS`] physical
    /// dims in another order than level-major.
    pub fn from_physical_dims(dims: &[impl AsRef<[usize]>]) -> Result<Tiling, MetadataError> {
        let fragments: SmallVec<[usize; MAX_LOGICAL_DIMS]> =
            dims.iter().map(|dim| dim.as_ref().len()).collect();
        let level_major = Tiling::new(&fragments)?;
        let rank = fragments.iter().sum();
        let mut order = PerPhysicalDim::from_elem(rank, rank);
        let sorted = level_major.physical_dims(dims.len());
        for (stated, sorted) in dims.iter().zip(&sorted) {
            for (&physical, &place) in stated.as_ref().iter().zip(sorted) {
                // A dim listed twice leaves another unlisted.
                if let Some(slot) = order.get_mut(physical) {
                    *slot = place;
                }
            }
        }
        let stated: SmallVec<[&[usize]; MAX_LOGICAL_DIMS]> =
            dims.iter().map(|dim| dim.as_ref()).collect();
        let rule = |holds: bool, name: &str| {
            holds
                .then_some(())
                .ok_or_else(|| invalid(format!("tiling over physical dims {stated:?}: {name}")))
        };
        rule(
            order.iter().all(|&place| place < rank),
            "each physical dim is one fragment",
        )?;
        if order.is_sorted() {
            return Ok(level_major);
        }
        rule(
            level_major.is_tiled(),
            "untiled dims are stored in their own order",
        )?;
        rule(
            rank <= MAX_ORDERED_DIMS,
            "at most 12 physical dims in another order than level-major",
        )?;
        Ok(level_major.ordered(place_of_order(&order)))
    }

    /// The number of fragments of each of `logical_rank` logical dims.
    pub fn fragments(self, logical_rank: usize) -> SmallVec<[usize; MAX_LOGICAL_DIMS]> {
        (0..logical_rank).map(|dim| self.levels(dim) + 1).collect()
    }

    /// The physical dims each of `logical_rank` logical dims is stored as.
    pub fn physical_dims(self, logical_rank: usize) -> PhysicalDims {
        let fragments = self.fragments(logical_rank);
        let order = order_at_place(self.place(), fragments.iter().sum());
        let mut physical = PerPhysicalDim::from_elem(0, order.len());
        for (dim, &place) in order.iter().enumerate() {
            physical[place] = dim;
        }
        let mut level_major = physical.into_iter();
        let mut dims = PhysicalDims::from_elem(SmallVec::new(), logical_rank);
        for level in 0..MAX_FRAGMENTS {
            for (dim, &count) in fragments.iter().enumerate() {
                if level < count {
                    dims[dim].extend(level_major.next());
                }
            }
        }
        dims
    }

    /// The logical rank of a buffer of `rank` physical dims: one per dim, less
    /// the extra fragments. Untiled, any rank, past [`MAX_LOGICAL_DIMS`] too.
    ///
    /// # Errors
    ///
    /// When `rank` holds fewer dims than the tiling's fragments or than its order lists, or a dim
    /// past the logical rank is tiled.
    pub fn logical_rank(self, rank: usize) -> Result<usize, MetadataError> {
        if !self.is_tiled() {
            return Ok(rank);
        }
        let extra: usize = (0..MAX_LOGICAL_DIMS).map(|dim| self.levels(dim)).sum();
        let ordered =
            |logical| Tiling::new(&self.fragments(logical)).map(|t| t.ordered(self.place()));
        let logical = rank
            .checked_sub(extra)
            .filter(|&logical| ordered(logical) == Ok(self) && is_an_order(self.place(), rank))
            .ok_or_else(|| invalid(format!("{self:?} does not fit a rank-{rank} buffer")))?;
        Ok(logical)
    }

    /// Whether any logical dim is stored as more than one fragment.
    pub fn is_tiled(self) -> bool {
        self.0 != 0
    }

    /// This tiling with logical dim `i` the one that was `axes[i]`, its physical dims in the same
    /// order relative to level-major.
    ///
    /// # Errors
    ///
    /// More than [`MAX_LOGICAL_DIMS`] axes.
    pub(crate) fn permuted(self, axes: &[usize]) -> Result<Tiling, MetadataError> {
        let fragments: SmallVec<[usize; MAX_LOGICAL_DIMS]> =
            axes.iter().map(|&axis| self.levels(axis) + 1).collect();
        Ok(Tiling::new(&fragments)?.ordered(self.place()))
    }

    /// The extra fragments of logical dim `dim`: none past the dims the low half holds, where the
    /// order's bits start.
    fn levels(self, dim: usize) -> usize {
        match dim < MAX_LOGICAL_DIMS {
            true => ((self.0 >> (4 * dim)) & 0b1111) as usize,
            false => 0,
        }
    }

    /// The place of the order its physical dims are listed in, `0` for level-major.
    fn place(self) -> u32 {
        (self.0 >> ORDER_SHIFT) as u32
    }

    /// This tiling, level-major, with its physical dims in the order at `place`.
    fn ordered(self, place: u32) -> Tiling {
        Tiling(self.0 | (place as u64) << ORDER_SHIFT)
    }
}

/// The place of `order`, at most [`MAX_ORDERED_DIMS`] distinct values, among every order of as
/// many, the sorted one first.
fn place_of_order(order: &[usize]) -> u32 {
    let mut place = 0;
    for (at, value) in order.iter().enumerate() {
        let after = &order[at + 1..];
        let smaller = after.iter().filter(|&other| other < value).count();
        place = place * (after.len() + 1) as u32 + smaller as u32;
    }
    place
}

/// The order of `0..len` at `place`, as [`place_of_order`] counts them: sorted for `0`, whatever
/// the length.
fn order_at_place(place: u32, len: usize) -> PerPhysicalDim {
    let mut place = place as usize;
    let mut smaller_after = PerPhysicalDim::from_elem(0, len);
    for at in (0..len).rev() {
        let remaining = len - at;
        smaller_after[at] = place % remaining;
        place /= remaining;
    }
    let mut values: PerPhysicalDim = (0..len).collect();
    smaller_after
        .into_iter()
        .map(|smaller| values.remove(smaller))
        .collect()
}

/// Whether `len` values have an order at `place`.
fn is_an_order(place: u32, len: usize) -> bool {
    let mut place = place as usize;
    for remaining in 1..=len {
        place /= remaining;
    }
    place == 0
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
        assert_eq!(Tiling::new(&[2, 2]).unwrap(), Tiling(0x11));
        assert_eq!(
            Tiling::new(&[16, 4]).unwrap().fragments(2).as_slice(),
            &[16, 4]
        );
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

    /// `[k, n]` stored `[k/32, n/64, 32, 64/8, 8]`: each dim's coarsest piece first, in logical
    /// order, then the next level of each dim still that deep.
    #[test]
    fn physical_dims_are_level_major() {
        let dims = Tiling::new(&[2, 3]).unwrap().physical_dims(2);
        assert_eq!(dims[0].as_slice(), &[0, 2]);
        assert_eq!(dims[1].as_slice(), &[1, 3, 4]);
        assert_eq!(Tiling::UNTILED.physical_dims(3)[2].as_slice(), &[2]);
    }

    /// An untiled tensor may have more dims than a tiling describes, and its logical rank is
    /// still its rank: every tensor's shape is read through it.
    #[test]
    fn an_untiled_buffer_of_any_rank_is_its_own_logical_rank() {
        assert_eq!(Tiling::UNTILED.logical_rank(9), Ok(9));
        assert_eq!(Tiling::UNTILED.fragments(9).as_slice(), &[1; 9]);
    }

    #[test]
    fn refuses_what_the_protocol_cannot_hold() {
        assert!(Tiling::new(&[1; 9]).is_err());
        assert!(Tiling::new(&[17]).is_err());
        assert!(Tiling::new(&[0]).is_err());
    }

    /// `[m, k]` stored `[m/16, k/16, m%8, k/2%4, k/8%2, m/8%2, k%2]`, as a tensor core takes a tile
    /// of sixteen by sixteen.
    const TENSOR_CORE: [&[usize]; 2] = [&[0, 5, 2], &[1, 4, 3, 6]];

    #[test]
    fn physical_dims_read_back_the_dims_stated() {
        let tiling = Tiling::from_physical_dims(&TENSOR_CORE).unwrap();
        let dims = tiling.physical_dims(2);
        assert_eq!(dims[0].as_slice(), TENSOR_CORE[0]);
        assert_eq!(dims[1].as_slice(), TENSOR_CORE[1]);
    }

    /// The value a level-major tiling is serialized and hashed as names no order.
    #[test]
    fn dims_stated_level_major_are_the_tiling_of_their_counts() {
        assert_eq!(
            Tiling::from_physical_dims(&[&[0, 2][..], &[1, 3, 4]]),
            Tiling::new(&[2, 3])
        );
        assert_eq!(Tiling::new(&[2, 3]), Ok(Tiling(0x21)));
    }

    /// Twelve dims listed back to front are the last order there is.
    #[test]
    fn the_last_order_of_twelve_dims_is_kept() {
        let dims: [&[usize]; 2] = [&[11, 9, 7, 5, 3, 1], &[10, 8, 6, 4, 2, 0]];
        let tiling = Tiling::from_physical_dims(&dims).unwrap();
        assert_eq!(tiling.place(), 479_001_599);
        assert_eq!(tiling.physical_dims(2)[0].as_slice(), dims[0]);
        assert_eq!(tiling.physical_dims(2)[1].as_slice(), dims[1]);
    }

    #[test]
    fn a_rank_its_order_does_not_list_is_refused() {
        let reversed = Tiling::from_physical_dims(&[&[4][..], &[3, 2, 1, 0]]).unwrap();
        assert_eq!(reversed.logical_rank(5), Ok(2));
        assert!(reversed.logical_rank(4).is_err());
    }

    #[test]
    fn refuses_physical_dims_that_are_not_each_one_fragment() {
        assert!(Tiling::from_physical_dims(&[&[0, 2][..], &[1, 1]]).is_err());
        assert!(Tiling::from_physical_dims(&[&[0, 4][..], &[1, 2]]).is_err());
    }

    #[test]
    fn refuses_untiled_dims_out_of_order() {
        assert!(Tiling::from_physical_dims(&[[1], [0]]).is_err());
    }

    #[test]
    fn refuses_more_ordered_dims_than_it_counts() {
        let thirteen: [&[usize]; 2] = [&[1, 0], &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]];
        assert!(Tiling::from_physical_dims(&thirteen).is_err());
    }
}
