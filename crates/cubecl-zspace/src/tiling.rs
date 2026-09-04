//! Storage tiling: which logical dim each physical dim is a fragment of.
//!
//! Strides say where a tensor's dims step and nothing about which of those dims
//! are fragments of one logical dim. A `[k, n]` matrix stored as
//! `[k/32, n/32, 32, 32]` has four dims and, to every stride-reading op, four
//! independent axes; only whoever tiled it knows that dims 0 and 2 together are
//! `k`. That is the one storage fact strides cannot express, and [`Tiling`]
//! carries it beside them.
//!
//! One label per physical dim, naming the logical dim it belongs to, packed two
//! bits a dim into a `u16`: eight physical dims, four logical ones. Which fragment
//! of a logical dim is the coarse one is not stored, because the strides already
//! say it, and the logical extent is the product of the fragments'. An untiled
//! tensor is the zero value.

use alloc::format;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::MetadataError;

/// The most physical dims a tiling describes: two bits each in a `u16`.
pub const MAX_TILED_DIMS: usize = 8;

/// The most logical dims a tiling names: the four values two bits can hold.
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
    /// More than [`MAX_TILED_DIMS`] dims; a label at or past [`MAX_TILED_LABELS`];
    /// labels that are not exactly `0..L` for some `L` (a logical dim nobody
    /// carries is a hole in the logical shape); or fewer than two logical dims,
    /// which is a reshape and not a tiling.
    pub fn new(labels: &[usize]) -> Result<Tiling, MetadataError> {
        if labels.len() > MAX_TILED_DIMS {
            return Err(invalid(format!(
                "a tiling describes at most {MAX_TILED_DIMS} physical dims, got {}",
                labels.len()
            )));
        }
        let mut seen = [false; MAX_TILED_LABELS];
        for &label in labels {
            if label >= MAX_TILED_LABELS {
                return Err(invalid(format!(
                    "a tiling names at most {MAX_TILED_LABELS} logical dims, got label {label}"
                )));
            }
            seen[label] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        if seen[..distinct].iter().any(|&s| !s) {
            return Err(invalid(format!(
                "a tiling's labels must be exactly 0..{distinct}, got {labels:?}"
            )));
        }
        if distinct < 2 {
            return Err(invalid(format!(
                "a tiling of one logical dim is a reshape, not a tiling: {labels:?}"
            )));
        }
        let mut bits = 0u16;
        for (dim, &label) in labels.iter().enumerate() {
            bits |= (label as u16) << (2 * dim);
        }
        Ok(Tiling(bits))
    }

    /// The label of each of `rank` physical dims, in buffer order. Meaningful
    /// for a tiled tiling; an untiled one labels every dim `0`.
    pub fn labels(self, rank: usize) -> SmallVec<[usize; MAX_TILED_DIMS]> {
        (0..rank)
            .map(|dim| ((self.0 >> (2 * dim)) & 0b11) as usize)
            .collect()
    }

    /// Whether any physical dim is a fragment of a logical one.
    pub fn is_tiled(self) -> bool {
        self.0 != 0
    }
}

fn invalid(message: alloc::string::String) -> MetadataError {
    MetadataError::Invalid { reason: message }
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
        assert!(!Tiling::UNTILED.is_tiled());
        assert_eq!(Tiling::default(), Tiling::UNTILED);
    }

    #[test]
    fn refuses_what_two_bits_a_dim_cannot_hold() {
        assert!(Tiling::new(&[0, 1, 0, 1, 0, 1, 0, 1, 0]).is_err());
        assert!(Tiling::new(&[0, 4]).is_err());
    }

    #[test]
    fn refuses_a_hole_and_a_single_logical_dim() {
        assert!(Tiling::new(&[0, 2]).is_err());
        assert!(Tiling::new(&[0, 0]).is_err());
    }
}
