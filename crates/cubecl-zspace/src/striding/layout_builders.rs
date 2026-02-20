//! # Stride Layout Builders

use crate::{Strides, strides};

/// Construct row-major contiguous strides for a shape.
///
/// This will return a new vec ``strides`` such that:
/// - ``rank == shape.len()``
/// - ``strides.len() == rank``
/// - ``strides[rank - 1] == 1``
/// - ``for i in 0..rank - 1 { strides[i] == strides[i + 1] * shape[i + 1] }``
///
/// If ``rank == 0``, this will return ``vec![]``.
pub fn row_major_contiguous_strides<S>(shape: S) -> Strides
where
    S: AsRef<[usize]>,
{
    let shape = shape.as_ref();
    let rank = shape.len();
    let mut strides = strides![1; rank];
    if rank > 1 {
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_contiguous_strides() {
        assert_eq!(row_major_contiguous_strides([]), strides![]);
        assert_eq!(row_major_contiguous_strides([1, 2, 3]), strides![6, 3, 1]);
    }
}
