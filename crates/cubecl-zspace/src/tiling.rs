//! Storage tiling: which logical dim each physical dim is a fragment of.
//!
//! Strides say where a tensor's dims step, not which dims are fragments of one
//! logical dim. A `[k, n]` matrix stored as `[k/32, n/32, 32, 32]` is four
//! independent axes to every stride-reading op; only whoever tiled it knows
//! that dims 0 and 2 together are `k`. [`Tiling`] carries that beside the
//! strides. The shape and strides stay physical.
//!
//! # Protocol
//!
//! A `u16`, two bits per physical dim, dim `i` at bits `2i..2i + 2`:
//!
//! ```text
//! bits  15 14 | 13 12 | 11 10 |  9 8  |  7 6  |  5 4  |  3 2  |  1 0
//! dim     7   |   6   |   5   |   4   |   3   |   2   |   1   |   0
//! ```
//!
//! Each field holds the dim's **label**: the logical dim it is a fragment of.
//!
//! | Rule | |
//! |---|---|
//! | Untiled is `0` | every physical dim is its own logical dim |
//! | At most 8 physical dims | two bits each in a `u16` |
//! | At most 4 logical dims | labels are `0..4` |
//! | Labels are exactly `0..L` | no logical dim nobody carries |
//! | Two or more logical dims | one logical dim's bits are all `0`, which reads as untiled |
//! | Coarse first | fragments of one logical dim are ordered by their strides; the logical extent is their product |
//!
//! `[k, n]` stored `[k/32, n/32, 32, 32]` has labels `[0, 1, 0, 1]`, which is
//! `0b01_00_01_00`. Labels no two dims share describe no fragments and are the
//! untiled value.

use alloc::format;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::MetadataError;

/// The most physical dims a tiling describes: two bits each in a `u16`.
pub const MAX_TILED_DIMS: usize = 8;

/// The most logical dims a tiling names: the four values two bits hold.
pub const MAX_TILED_LABELS: usize = 4;

/// Which logical dim each physical dim is a fragment of. See the [module doc](self).
///
/// `Default` is untiled, and so is `0`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Tiling(u16);

impl Tiling {
    /// Every physical dim is its own logical dim.
    pub const UNTILED: Tiling = Tiling(0);

    /// The tiling whose physical dim `i` is a fragment of logical dim `labels[i]`.
    ///
    /// # Errors
    ///
    /// A label set the protocol cannot hold: see the [rules](self).
    pub fn new(labels: &[usize]) -> Result<Tiling, MetadataError> {
        let logical = labels.iter().max().map_or(0, |max| max + 1);
        let rule = |holds: bool, name: &str| {
            holds
                .then_some(())
                .ok_or_else(|| invalid(format!("tiling {labels:?}: {name}")))
        };
        rule(labels.len() <= MAX_TILED_DIMS, "at most 8 physical dims")?;
        rule(logical <= MAX_TILED_LABELS, "at most 4 logical dims")?;
        rule(
            (0..logical).all(|label| labels.contains(&label)),
            "labels are exactly 0..L",
        )?;
        if logical == labels.len() {
            return Ok(Tiling::UNTILED);
        }
        rule(logical >= 2, "two or more logical dims")?;

        let field = |(dim, &label): (usize, &usize)| (label as u16) << (2 * dim);
        Ok(Tiling(labels.iter().enumerate().map(field).sum()))
    }

    /// The label of each of `rank` physical dims, in buffer order; `0..rank`
    /// when untiled.
    pub fn labels(self, rank: usize) -> SmallVec<[usize; MAX_TILED_DIMS]> {
        match self.is_tiled() {
            true => (0..rank).map(|dim| self.label(dim)).collect(),
            false => (0..rank).collect(),
        }
    }

    /// Whether any physical dim is a fragment of a logical one.
    pub fn is_tiled(self) -> bool {
        self.0 != 0
    }

    fn label(self, dim: usize) -> usize {
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
    fn labels_round_trip() {
        let labels = [0, 1, 2, 1, 2];
        let tiling = Tiling::new(&labels).unwrap();
        assert!(tiling.is_tiled());
        assert_eq!(tiling.labels(5).as_slice(), &labels);
        assert_eq!(Tiling::new(&[0, 1, 0, 1]).unwrap(), Tiling(0b01_00_01_00));
    }

    #[test]
    fn labels_no_two_dims_share_are_untiled() {
        assert_eq!(Tiling::new(&[0, 1, 2]).unwrap(), Tiling::UNTILED);
        assert_eq!(Tiling::new(&[]).unwrap(), Tiling::UNTILED);
        assert_eq!(Tiling::UNTILED.labels(3).as_slice(), &[0, 1, 2]);
        assert_eq!(Tiling::default(), Tiling::UNTILED);
    }

    #[test]
    fn refuses_what_the_protocol_cannot_hold() {
        assert!(Tiling::new(&[0, 1, 0, 1, 0, 1, 0, 1, 0]).is_err());
        assert!(Tiling::new(&[0, 4, 0]).is_err());
        assert!(Tiling::new(&[0, 2, 0]).is_err());
        assert!(Tiling::new(&[0, 0]).is_err());
    }
}
