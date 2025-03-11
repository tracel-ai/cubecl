use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Layout for matrix tensors, i.e. tensors whose interpretation
/// is a bunch of batched matrices of 2 dimensions
pub enum MatrixLayout {
    /// Memory is wholly contiguous, with row major layout
    Contiguous,
    /// Permutations happened, but may not impact some kernels
    MildlyPermuted {
        /// Last two dims are inverted
        transposed: bool,
        /// Some permutations exist in batch dimensions
        batch_swap: bool,
    },
    /// Permutations happened between batch dimensions and last two dims
    HighlyPermuted,
}

/// Return the layout of a matrix given the strides.
pub fn matrix_layout(strides: &[usize]) -> MatrixLayout {
    let rank = strides.len();
    if rank <= 1 {
        return MatrixLayout::Contiguous;
    }

    let mut transposed = false;
    let mut batch_swap = false;
    let row_stride = strides[rank - 2];
    let col_stride = strides[rank - 1];
    if row_stride == 0 || col_stride == 0 {
        // Broadcasted last two dims
        return MatrixLayout::HighlyPermuted;
    }
    if row_stride < col_stride {
        transposed = true;
    }
    let mut previous_stride = row_stride;

    for d in 0..rank - 2 {
        let current_stride = strides[rank - 3 - d];
        if current_stride < row_stride || current_stride < col_stride {
            if current_stride == 0 {
                // Broadcasted batch dim
                batch_swap = true;
            } else {
                return MatrixLayout::HighlyPermuted;
            }
        }
        if current_stride < previous_stride {
            batch_swap = true;
        }

        previous_stride = current_stride;
    }

    if transposed || batch_swap {
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        }
    } else {
        MatrixLayout::Contiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_contiguous() {
        let strides = &[8, 4, 2, 1];
        assert_eq!(matrix_layout(strides), MatrixLayout::Contiguous);
    }

    #[test]
    fn vector_is_contiguous() {
        let strides = &[1];
        assert_eq!(matrix_layout(strides), MatrixLayout::Contiguous)
    }

    #[test]
    fn layout_is_transposed_only() {
        let strides = &[8, 4, 1, 2];
        if let MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = matrix_layout(strides)
        {
            assert!(transposed && !batch_swap);
        } else {
            unreachable!()
        }
    }

    #[test]
    fn layout_has_swapped_batches_only() {
        let strides = &[4, 8, 2, 1];
        if let MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = matrix_layout(strides)
        {
            assert!(!transposed && batch_swap);
        } else {
            unreachable!()
        }
    }

    #[test]
    fn layout_has_swapped_batches_and_is_transposed() {
        let strides = &[4, 8, 1, 2];
        if let MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = matrix_layout(strides)
        {
            assert!(transposed && batch_swap);
        } else {
            unreachable!()
        }
    }

    #[test]
    fn layout_has_batch_swapped_with_row() {
        let strides = &[8, 2, 4, 1];
        assert_eq!(matrix_layout(strides), MatrixLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_batch_swapped_with_col() {
        let strides = &[1, 4, 2, 8];
        assert_eq!(matrix_layout(strides), MatrixLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_multiple_broadcasted_dims() {
        // E.g., tensor w/ shape [1, 4] expanded to [2, 3, 4]
        let strides = &[0, 0, 1];
        assert_eq!(matrix_layout(strides), MatrixLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_row_broadcasted() {
        // E.g., tensor w/ shape [1, 4] expanded to [3, 4]
        let strides = &[0, 1];
        assert_eq!(matrix_layout(strides), MatrixLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_col_broadcasted() {
        // E.g., tensor w/ shape [2, 1] expanded to [2, 3]
        let strides = &[1, 0];
        assert_eq!(matrix_layout(strides), MatrixLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_batch_broadcasted() {
        // E.g., tensor w/ shape [2, 4] expanded to [2, 2, 4]
        let strides = &[0, 4, 1];
        if let MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = matrix_layout(strides)
        {
            assert!(!transposed && batch_swap);
        } else {
            unreachable!()
        }
    }

    #[test]
    fn layout_has_multiple_batch_broadcasted() {
        // E.g., tensor w/ shape [2, 4] expanded to [2, 2, 2, 4]
        let strides = &[0, 0, 4, 1];
        if let MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = matrix_layout(strides)
        {
            assert!(!transposed && batch_swap);
        } else {
            unreachable!()
        }
    }
}
