//! # Stride Layout Utilities

use alloc::vec::Vec;
use core::error::Error;
use core::fmt::{Display, Formatter};

/// Collected shape/stride record.
///
/// As this is used for error messages, there is no expectation that this is valid,
/// or that the ranks match.
#[derive(Debug, Clone, PartialEq)]
pub struct StrideRecord {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
}

impl StrideRecord {
    /// Create a new StrideRecord from a slice of usize strides.
    pub fn from_usize_strides(shape: &[usize], strides: &[usize]) -> StrideRecord {
        StrideRecord {
            shape: shape.to_vec(),
            strides: strides.iter().map(|s| *s as isize).collect(),
        }
    }

    /// Create a new StrideRecord from a slice of isize strides.
    pub fn from_isize_strides(shape: &[usize], strides: &[isize]) -> StrideRecord {
        StrideRecord {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        }
    }
}

/// Error describing striding issues.
#[derive(Debug, Clone, PartialEq)]
pub enum StrideError {
    /// The ranks of the shape and strides do not match.
    MalformedRanks { record: StrideRecord },

    /// This is an unsupported rank.
    UnsupportedRank { rank: usize, record: StrideRecord },

    /// The strides violate a constraint.
    Invalid {
        message: String,
        record: StrideRecord,
    },
}

impl Display for StrideError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            StrideError::MalformedRanks { record } => write!(f, "Malformed strides: {:?}", record),
            StrideError::UnsupportedRank { rank, record } => {
                write!(f, "Unsupported rank {}: {:?}", rank, record)
            }
            StrideError::Invalid { message, record } => {
                write!(f, "Invalid strides: {}: {:?}", message, record)
            }
        }
    }
}

impl Error for StrideError {}

/// Validate that a `shape`/`stride` pair has matching ranks.
///
/// # Arguments
/// * `shape` - the shape of a tensor.
/// * `strides` - the skip-strides of a tensor.
///
/// # Returns
/// `Ok(rank)` if the ranks match, otherwise `Err(StrideError::MalformedRanks)`
pub fn try_check_matching_ranks<A, B>(shape: A, strides: B) -> Result<usize, StrideError>
where
    A: AsRef<[usize]>,
    B: AsRef<[usize]>,
{
    let shape = shape.as_ref();
    let strides = strides.as_ref();

    let rank = shape.len();
    if strides.len() != rank {
        Err(StrideError::MalformedRanks {
            record: StrideRecord::from_usize_strides(shape, strides),
        })
    } else {
        Ok(rank)
    }
}

/// Validate that a `shape`/`stride` pair is row-major and non-zero on all dimensions.
///
/// # Arguments
/// * `shape` - the shape of a tensor.
/// * `strides` - the skip-strides of a tensor.
///
/// # Returns
/// * `Ok(())` - if the strides are non-zero and row-major,
/// * `Err(StrideError::MalformedRanks)` - if the ranks do not match,
/// * `Err(StrideError::UnsupportedRank)` - if the rank is 0,
pub fn try_check_pitched_row_major_strides<A, B>(shape: A, strides: B) -> Result<(), StrideError>
where
    A: AsRef<[usize]>,
    B: AsRef<[usize]>,
{
    let shape = shape.as_ref();
    let strides = strides.as_ref();

    let rank = try_check_matching_ranks(shape, strides)?;

    if rank == 0 {
        return Err(StrideError::UnsupportedRank {
            rank,
            record: StrideRecord::from_usize_strides(shape, strides),
        });
    }

    let mut valid_layout = strides[rank - 1] == 1 && strides.iter().all(|s| *s != 0);
    if valid_layout && rank > 1 {
        if strides[rank - 2] < shape[rank - 1] {
            valid_layout = false;
        }
        for i in 0..rank - 2 {
            if strides[i] != shape[i + 1] * strides[i + 1] {
                valid_layout = false;
                break;
            }
        }
    }

    if valid_layout {
        Ok(())
    } else {
        Err(StrideError::Invalid {
            message: "strides are not valid pitched row major order".to_string(),
            record: StrideRecord::from_usize_strides(shape, strides),
        })
    }
}

/// Check that the shape/stride layout is valid for cubecl layout.
///
/// # Returns
///
/// `true` if the shape and strides are valid for cubecl layout, `false` otherwise.
///
/// # Panics
/// - if `shape.len() == 0`.
/// - If `shape.len() != strides.len()`.
pub fn has_pitched_row_major_strides<A, B>(shape: A, strides: B) -> bool
where
    A: AsRef<[usize]>,
    B: AsRef<[usize]>,
{
    // TODO: migrate call sites to the `try_..()` form.
    // This contract (bool for some things, panic for others)
    // is a continuation of legacy code,

    match try_check_pitched_row_major_strides(shape, strides) {
        Ok(()) => true,
        Err(err) => match err {
            StrideError::UnsupportedRank { .. } | StrideError::MalformedRanks { .. } => {
                panic!("{err}")
            }
            StrideError::Invalid { .. } => false,
        },
    }
}

/// Validate that a `shape`/`stride` pair is contiguous and row-major.
///
/// # Arguments
/// * `shape` - the shape of a tensor.
/// * `strides` - the skip-strides of a tensor.
///
/// # Returns
/// * `Ok(())` - if the strides are contiguous and row-major,
/// * `Err(StrideError::MalformedRanks)` - if the ranks do not match,
/// * `Err(StrideError::UnsupportedRank)` - if the rank is 0,
pub fn try_check_contiguous_row_major_strides<A, B>(shape: A, strides: B) -> Result<(), StrideError>
where
    A: AsRef<[usize]>,
    B: AsRef<[usize]>,
{
    let shape = shape.as_ref();
    let strides = strides.as_ref();

    let rank = try_check_matching_ranks(shape, strides)?;

    if rank == 0 {
        return Err(StrideError::UnsupportedRank {
            rank,
            record: StrideRecord::from_usize_strides(shape, strides),
        });
    }

    let mut valid_layout = strides[rank - 1] == 1;
    if valid_layout && rank > 1 {
        for i in 0..rank - 1 {
            if strides[i] != shape[i + 1] * strides[i + 1] {
                valid_layout = false;
                break;
            }
        }
    }
    if valid_layout {
        Ok(())
    } else {
        Err(StrideError::Invalid {
            message: "strides are not contiguous in row major order".to_string(),
            record: StrideRecord::from_usize_strides(shape, strides),
        })
    }
}

/// Check that the shape/stride layout is contiguous
///
/// # Returns
///
/// `true` if the shape and strides are contiguous, `false` otherwise.
///
/// # Panics
/// - if `shape.len() == 0`.
/// - If `shape.len() != strides.len()`.
pub fn has_contiguous_row_major_strides<A, B>(shape: A, strides: B) -> bool
where
    A: AsRef<[usize]>,
    B: AsRef<[usize]>,
{
    // TODO: migrate call sites to the `try_..()` form.
    // This contract (bool for some things, panic for others)
    // is a continuation of legacy code,

    match try_check_contiguous_row_major_strides(shape, strides) {
        Ok(()) => true,
        Err(err) => match err {
            StrideError::UnsupportedRank { .. } | StrideError::MalformedRanks { .. } => {
                panic!("{err}")
            }
            StrideError::Invalid { .. } => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_check_matching_ranks() {
        assert_eq!(try_check_matching_ranks([1, 2, 3], [1, 2, 3]).unwrap(), 3);

        assert_eq!(
            &try_check_matching_ranks([1, 2], [1, 2, 3]),
            &Err(StrideError::MalformedRanks {
                record: StrideRecord {
                    shape: vec![1, 2],
                    strides: vec![1, 2, 3]
                }
            })
        );
    }

    #[test]
    fn test_try_check_contiguous_row_major_strides() {
        try_check_contiguous_row_major_strides([0], [1]).unwrap();
        try_check_contiguous_row_major_strides([2], [1]).unwrap();
        try_check_contiguous_row_major_strides([3, 2], [2, 1]).unwrap();
        try_check_contiguous_row_major_strides([4, 3, 2], [6, 2, 1]).unwrap();

        // rank=0
        assert_eq!(
            try_check_contiguous_row_major_strides([], []),
            Err(StrideError::UnsupportedRank {
                rank: 0,
                record: StrideRecord {
                    shape: vec![],
                    strides: vec![]
                }
            })
        );

        // non-contiguous
        assert_eq!(
            try_check_contiguous_row_major_strides([2, 2], [3, 1]),
            Err(StrideError::Invalid {
                message: "strides are not contiguous in row major order".to_string(),
                record: StrideRecord {
                    shape: vec![2, 2],
                    strides: vec![3, 1]
                }
            })
        );

        // not row-major
        assert_eq!(
            try_check_contiguous_row_major_strides([1, 2], [1, 2]),
            Err(StrideError::Invalid {
                message: "strides are not contiguous in row major order".to_string(),
                record: StrideRecord {
                    shape: vec![1, 2],
                    strides: vec![1, 2]
                }
            })
        );
    }

    #[test]
    #[should_panic]
    fn test_has_contiguous_row_major_strides_malformed_ranks() {
        has_contiguous_row_major_strides([1, 2], [1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_has_contiguous_row_major_strides_unsupported_rank() {
        has_contiguous_row_major_strides([], []);
    }

    #[test]
    fn test_has_contiguous_row_major_strides() {
        assert!(has_contiguous_row_major_strides([0], [1]));
        assert!(has_contiguous_row_major_strides([2], [1]));
        assert!(has_contiguous_row_major_strides([3, 2], [2, 1]));
        assert!(has_contiguous_row_major_strides([4, 3, 2], [6, 2, 1]));

        // non-contiguous
        assert!(!has_contiguous_row_major_strides([1], [2]));

        // not row-major
        assert!(!has_contiguous_row_major_strides([1, 2], [1, 2]));
    }
}
