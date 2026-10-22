//! A buffer layout's alignment in elements, along its innermost dim.

use crate::{Axis, Geometry, layout::StoragePartitioning};
use cubecl::prelude::TensorBinding;

/// The alignment of a buffer's layout, counted in elements along its innermost dim: the number
/// every line width that reads the buffer's elements and no others divides.
///
/// Not a byte or address alignment: the tile launcher refuses a width that does not divide the
/// innermost extent or any coarser stride, and every width past one where the innermost stride is
/// not one. Those are all "divides", so the layout is aligned to their greatest common divisor, and
/// that is the whole of what this holds, read once off the geometry so a derivation can ask it
/// without the tensor. It is one of three things a bound width answers to: the device's ceiling,
/// this, and a kernel's own requirements ([`Vectorization`](crate::launch::Vectorization)).
///
/// A view's start is the one address it reads: a view starting partway into its buffer is aligned
/// to its offset in elements ([`at_start`](Self::at_start)), the buffer's own start being the
/// runtime's to align.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElemAlignment {
    divides: usize,
    /// Whether `divides` is the one width that reads the buffer rather than a bound every width
    /// divides: a storage-tiled buffer stores its read as a piece, which a reader takes whole.
    exact: bool,
}

impl ElemAlignment {
    /// A layout aligned to one element, read one element at a time: a strided or broadcast view,
    /// whose innermost dim strides past one.
    pub const ONE: Self = Self {
        divides: 1,
        exact: false,
    };

    /// The alignment of a buffer of `shape` at `strides`, its dims taken in the order the buffer
    /// steps them, whatever order the view names them in.
    pub fn in_stride_order(shape: &[usize], strides: &[usize]) -> Self {
        let mut dims: Vec<(usize, usize)> =
            shape.iter().copied().zip(strides.iter().copied()).collect();
        dims.sort_by_key(|&(_, stride)| core::cmp::Reverse(stride));
        let (shape, strides): (Vec<usize>, Vec<usize>) = dims.into_iter().unzip();
        Self::new(&shape, &strides)
    }

    /// The alignment of rows of `columns` elements laid one after another: a buffer a launch
    /// allocates or copies dense. Its rows count for nothing, as only the step between them does.
    pub fn dense_rows(columns: usize) -> Self {
        Self::new(&[columns], &[1])
    }

    /// The alignment of a buffer of `shape` at `strides`, innermost last.
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        let divides = match (shape.split_last(), strides.split_last()) {
            (Some((&extent, _)), Some((&1, coarser))) => {
                coarser.iter().fold(extent, |acc, &stride| gcd(acc, stride))
            }
            _ => 1,
        };
        Self {
            divides,
            exact: false,
        }
    }

    /// The alignment a view starting `start_bytes` into its buffer gives lines of `element_bytes`
    /// elements: the start's offset in elements, which every line's first element sits a whole
    /// number of lines past, so a width dividing it keeps each line on its own boundary. A start
    /// at no whole element is aligned to one; a view at its buffer's start, to any width.
    pub fn at_start(start_bytes: u64, element_bytes: usize) -> Self {
        let element_bytes = element_bytes.max(1) as u64;
        let divides = match start_bytes.is_multiple_of(element_bytes) {
            true => (start_bytes / element_bytes) as usize,
            false => 1,
        };
        Self {
            divides,
            exact: false,
        }
    }

    /// The alignment of a buffer stored in pieces `read` elements wide, which a reader takes whole:
    /// a storage-tiled one's.
    pub const fn stored_piece(read: usize) -> Self {
        Self {
            divides: read,
            exact: true,
        }
    }

    /// Whether the buffer is read in the pieces it is stored in, a storage-tiled one's, rather than
    /// along rows.
    pub fn is_stored_piece(self) -> bool {
        self.exact
    }

    /// Whether a line `width` wide reads this buffer's elements and no others.
    pub fn admits(self, width: usize) -> bool {
        match self.exact {
            _ if width == 1 => true,
            true => width == self.divides,
            false => self.divides.is_multiple_of(width),
        }
    }
}

impl From<&TensorBinding> for ElemAlignment {
    /// The alignment of a bound matrix operand: its geometry's, or, for a storage-tiled one, the
    /// piece its storage partitioning states as the finest along its columns, and a piece one
    /// element wide where none does: no partitioning describes it, or its tiles step down a
    /// column first, which the matmul kernels refuse before reading this.
    fn from(binding: &TensorBinding) -> Self {
        if !binding.tiling.is_tiled() {
            return Self::new(&binding.shape, &binding.strides);
        }
        const ROWS: Axis = Axis(0);
        const COLS: Axis = Axis(1);
        let geometry = Geometry::from(binding);
        let storage = StoragePartitioning::new(&geometry, &geometry.labels(&[ROWS, COLS]));
        match storage.as_ref().and_then(|storage| storage.tiles().first()) {
            Some(&(COLS, read)) => Self::stored_piece(read),
            _ => Self::stored_piece(1),
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

#[cfg(test)]
mod tests {
    /// A col weight's rows run along `k`, a pitch apart, whichever way the view names them.
    #[test]
    fn a_view_is_aligned_in_the_order_its_buffer_steps() {
        let (k, n, pitch) = (64, 48, 72);
        let col = ElemAlignment::in_stride_order(&[4, k, n], &[n * pitch, 1, pitch]);
        assert!(col.admits(8) && !col.admits(16));
        assert!(ElemAlignment::in_stride_order(&[4, k, n], &[k * n, n, 1]).admits(16));
    }

    use super::*;

    /// A dense row is aligned to its length; padding past it to what the pad
    /// divides too; a strided innermost dim to one element.
    /// A view's start aligns it to its offset in elements: any width at its buffer's start, the
    /// offset's divisors partway in, one element where the start splits an element.
    #[test]
    fn a_start_is_aligned_to_its_offset_in_elements() {
        assert!(ElemAlignment::at_start(0, 2).admits(16));
        let partway = ElemAlignment::at_start(24, 2);
        assert!(partway.admits(4) && !partway.admits(8));
        assert!(!ElemAlignment::at_start(3, 2).admits(2));
    }

    #[test]
    fn a_layout_is_aligned_to_the_divisors_of_its_extent_and_strides() {
        let dense = ElemAlignment::new(&[4, 1500], &[1500, 1]);
        assert!(dense.admits(4) && !dense.admits(8));
        let pitched = ElemAlignment::new(&[4, 1504], &[1508, 1]);
        assert!(pitched.admits(4) && !pitched.admits(8));
        let col = ElemAlignment::new(&[4096, 4096], &[1, 4096]);
        assert!(col.admits(1) && !col.admits(2));
        let broadcast = ElemAlignment::new(&[8, 64], &[0, 1]);
        assert!(broadcast.admits(64));
    }
}
