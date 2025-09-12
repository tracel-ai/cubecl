//! Stride compatibility helpers for preflight checks and host I/O planning.
//!
//! These utilities describe common stride patterns independently of any backend,
//! so hosts and higher layers can make informed choices and surface clearer errors.
//!
//! Strides are expressed in element units (not bytes). Element size may be used
//! by callers to convert to/from byte pitches as needed.

use crate::server::AllocationKind;
use alloc::vec;
use alloc::vec::Vec;

/// Canonical contiguous row-major strides for a given shape (in elements).
///
/// Example: shape [R, C] -> strides [C, 1]
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    let mut s = 1usize;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = s;
        s = s.saturating_mul(*dim.max(&1));
    }
    strides
}

/// A coarse description of a stride pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridePattern {
    /// Fully contiguous row-major layout.
    Contiguous,
    /// 2D with inner-most contiguous axis and a row pitch (in elements) on the outer axis.
    /// `row_pitch_elems >= cols` is required.
    InnerContiguous2D {
        /// Pitch between consecutive rows in elements (not bytes).
        row_pitch_elems: usize,
    },
    /// Rank >= 2 with inner-most contiguous axis and a row pitch over all outer dimensions flattened
    /// into rows. `row_pitch_elems >= cols` is required.
    InnerContiguousRows {
        /// Pitch between consecutive flattened rows in elements (not bytes).
        row_pitch_elems: usize,
    },
    /// Any other non-supported or irregular stride pattern.
    Other,
}

/// Describe the given shape/strides pair.
pub fn describe(shape: &[usize], strides: &[usize]) -> StridePattern {
    if shape.len() != strides.len() {
        return StridePattern::Other;
    }

    if strides == contiguous_strides(shape).as_slice() {
        return StridePattern::Contiguous;
    }

    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let row_pitch = strides[0];
        let inner = strides[1];

        // Accept inner-most contiguous 2D with row pitch >= cols.
        if inner == 1 && row_pitch >= cols && rows > 0 && cols > 0 {
            return StridePattern::InnerContiguous2D {
                row_pitch_elems: row_pitch,
            };
        }
    }

    // General inner-contiguous rows for rank >= 2: last axis contiguous, outer strides chain
    // multiplicatively while allowing row pitch padding on the last-but-one axis.
    if shape.len() >= 2 {
        let last = shape.len() - 1;
        if strides[last] == 1 {
            // Verify the stride chain for the outer dimensions: s[i] == shape[i+1] * s[i+1]
            let mut ok = true;
            for i in 0..last - 1 {
                if strides[i] != shape[i + 1].saturating_mul(strides[i + 1]) {
                    ok = false;
                    break;
                }
            }
            // For rank==2, the above loop is skipped; we fall back to the >= cols check below
            let row_pitch = strides[last - 1];
            if ok && row_pitch >= shape[last] {
                return StridePattern::InnerContiguousRows {
                    row_pitch_elems: row_pitch,
                };
            }
        }
    }

    StridePattern::Other
}

/// Whether the given shape/strides is fully contiguous.
#[inline]
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    matches!(describe(shape, strides), StridePattern::Contiguous)
}

/// Whether the given shape/strides is rank-2 with inner-most contiguous axis and a row pitch.
#[inline]
pub fn is_inner_contiguous_2d(shape: &[usize], strides: &[usize]) -> bool {
    matches!(
        describe(shape, strides),
        StridePattern::InnerContiguous2D { .. }
    )
}

/// Whether the given shape/strides is rank>=2 with inner-most contiguous axis and a row pitch
/// across all outer dimensions flattened.
#[inline]
pub fn is_inner_contiguous_rows(shape: &[usize], strides: &[usize]) -> bool {
    matches!(
        describe(shape, strides),
        StridePattern::InnerContiguous2D { .. } | StridePattern::InnerContiguousRows { .. }
    )
}

/// If `shape/strides` forms inner-contiguous rows (rank>=2), return the row pitch (in elements).
#[inline]
pub fn row_pitch_elems(shape: &[usize], strides: &[usize]) -> Option<usize> {
    match describe(shape, strides) {
        StridePattern::InnerContiguous2D { row_pitch_elems }
        | StridePattern::InnerContiguousRows { row_pitch_elems } => Some(row_pitch_elems),
        _ => None,
    }
}

/// Compute pitched-rows layout and allocation size for rank>1 tensors.
/// Returns (strides_in_elements, total_size_in_bytes). The row pitch (in bytes)
/// is aligned up to `align`.
pub fn pitched_rows_layout(shape: &[usize], elem_size: usize, align: usize) -> (Vec<usize>, usize) {
    let rank = shape.len();
    let width = *shape.last().unwrap_or(&1);
    let height: usize = shape.iter().rev().skip(1).product();
    let height = height.max(1);
    let width_bytes = width * elem_size;
    let row_pitch_bytes = width_bytes.next_multiple_of(align);
    let size = height * row_pitch_bytes;

    let mut strides = vec![1usize; rank];
    if rank > 1 {
        strides[rank - 2] = row_pitch_bytes / elem_size;
    }
    if rank > 2 {
        for i in (0..rank - 2).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    (strides, size)
}

/// Suggest an allocation kind given a shape/strides pair.
/// - Contiguous -> `AllocationKind::Contiguous`
/// - Inner-contiguous rows -> `AllocationKind::Optimized`
/// - Otherwise fallback to `AllocationKind::Contiguous` (caller may still reject on use).
pub fn preferred_allocation_kind(shape: &[usize], strides: &[usize]) -> AllocationKind {
    if is_contiguous(shape, strides) {
        AllocationKind::Contiguous
    } else if is_inner_contiguous_rows(shape, strides) {
        AllocationKind::Optimized
    } else {
        AllocationKind::Contiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_for_1d_and_2d() {
        assert_eq!(contiguous_strides(&[8]), vec![1]);
        assert_eq!(contiguous_strides(&[2, 3]), vec![3, 1]);
        assert!(is_contiguous(&[8], &[1]));
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_contiguous(&[2, 3], &[4, 1]));
    }

    #[test]
    fn inner_contiguous_2d_detection() {
        // 2D pitched: rows=4, cols=5, pitch=8 (in elems)
        let shape = [4, 5];
        let strides = [8, 1];
        assert!(is_inner_contiguous_2d(&shape, &strides));
        match describe(&shape, &strides) {
            StridePattern::InnerContiguous2D { row_pitch_elems } => assert_eq!(row_pitch_elems, 8),
            other => panic!("unexpected: {other:?}"),
        }
        // Not inner-contiguous
        assert!(!is_inner_contiguous_2d(&shape, &[8, 2]));
        // Pitch less than cols should not be accepted
        assert!(!is_inner_contiguous_2d(&shape, &[4, 1]));
    }

    #[test]
    fn inner_contiguous_rows_rank3() {
        // Rank 3 with inner-contiguous rows and padded pitch
        let shape = [2, 3, 5]; // rows=2*3=6, cols=5
        let row_pitch = 8usize;
        // Stride chain: s[1] arbitrary row_pitch, s[0] == shape[1] * s[1] == 3 * 8 = 24
        let strides = [24, row_pitch, 1];
        assert!(is_inner_contiguous_rows(&shape, &strides));
        assert_eq!(row_pitch_elems(&shape, &strides), Some(row_pitch));
        // Reject when last stride not 1
        assert!(!is_inner_contiguous_rows(&shape, &[24, 8, 2]));
        // Reject when chain breaks
        assert!(!is_inner_contiguous_rows(&shape, &[16, 8, 1]));
    }

    #[test]
    fn describe_other() {
        // Rank 3 non-contiguous pattern should be Other
        assert!(matches!(
            describe(&[2, 3, 4], &[10, 4, 1]),
            StridePattern::Other
        ));
        // Mismatched lengths
        assert!(matches!(describe(&[2], &[]), StridePattern::Other));
    }
}
