//! Which of a space's axes form the matrix a 2-D reader sees.

use core::fmt::{self, Display, Formatter};

use crate::{Extent, Space};

/// Which of a tile's axes form a 2-D matrix: a batch prefix, then the row group, then the
/// column group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatrixAxes {
    /// Where the row group starts; everything before it is batch.
    pub row_split: usize,
    /// Where the column group starts.
    pub col_split: usize,
}

impl MatrixAxes {
    /// The trailing pair: leading axes batch, the last two the matrix.
    pub(crate) fn trailing(space: &Space) -> Self {
        let rank = space.rank();
        MatrixAxes {
            row_split: rank - 2,
            col_split: rank - 1,
        }
    }

    /// The innermost axis as columns and the last axis above it of extent past one as rows.
    pub(crate) fn edges(space: &Space) -> Self {
        let rank = space.rank();
        let col_split = rank - 1;
        // A dynamic axis is not a number one.
        let row_split = (0..col_split)
            .rev()
            .find(|&p| !matches!(space.extent_raw(space.axis_at(p)), Extent::Static(1)))
            .unwrap_or(col_split.saturating_sub(1));
        MatrixAxes {
            row_split,
            col_split,
        }
    }

    /// An operand's matrix against the output `out` it contracts into: the axes `out` lacks as
    /// one edge, however many axes make up the contraction, and the free axis beside them as the
    /// other. The contraction is the window's trailing run of axes `out` lacks, or the run below
    /// its free trailing axis.
    pub(crate) fn operand(window: &Space, out: &Space) -> Self {
        let rank = window.rank();
        let contracted = |p: usize| !out.contains(window.axis_at(p));
        // The run of contracted axes ending at `end`, exclusive: its first position.
        let run_start = |end: usize| {
            let mut start = end;
            while start > 0 && contracted(start - 1) {
                start -= 1;
            }
            start
        };
        match contracted(rank - 1) {
            // `{.., free, k..}`: the contraction the columns, the free axis above it the rows.
            true => {
                let col_split = run_start(rank);
                MatrixAxes {
                    row_split: col_split.saturating_sub(1),
                    col_split,
                }
            }
            // `{.., k.., free}`: the free axis the columns, the contraction above it the rows.
            false => MatrixAxes {
                row_split: run_start(rank - 1),
                col_split: rank - 1,
            },
        }
    }

    /// An accumulator's matrix, against the lhs it is contracted with.
    pub(crate) fn accumulator(acc: &Space, lhs: &Space) -> Self {
        let mut col_split = acc.rank() - 1;
        while col_split > 1 && !lhs.contains(acc.axis_at(col_split - 1)) {
            col_split -= 1;
        }
        MatrixAxes {
            row_split: col_split - 1,
            col_split,
        }
    }

    /// The row edge these axes give in `space`.
    pub(crate) fn rows(&self, space: &Space) -> usize {
        (self.row_split..self.col_split)
            .map(|p| space.extent_at(p))
            .product()
    }

    /// The column edge, in scalars.
    pub(crate) fn cols(&self, space: &Space) -> usize {
        (self.col_split..space.rank())
            .map(|p| space.extent_at(p))
            .product()
    }

    /// A run of `values` scalars along the columns from their origin, as an extent per axis of
    /// `space`: the innermost axes taken whole, outwards, until the run is covered, every other
    /// axis one. Over a `k` split in blocks of 16, a run of 32 is the whole of the position inside
    /// a block, two blocks along.
    ///
    /// # Panics
    ///
    /// Where the column axes cover no run of `values` whole.
    pub(crate) fn col_run(&self, space: &Space, values: usize) -> Vec<usize> {
        let mut run = vec![1; space.rank()];
        let mut left = values;
        for p in (self.col_split..space.rank()).rev() {
            let taken = space.extent_at(p).min(left);
            assert!(
                left.is_multiple_of(taken),
                "MatrixAxes::col_run: a run of {values} values along the columns, which these \
                 axes do not cover whole"
            );
            run[p] = taken;
            left /= taken;
        }
        run
    }

    /// The axes giving a `rows x cols` matrix, both scalar, found from the innermost axis outwards.
    pub(crate) fn new(space: &Space, rows: usize, cols: usize) -> Result<Self, NoMatrix> {
        let rank = space.rank();
        let mut col_split = rank;
        let mut trailing = 1;
        while col_split > 0 && trailing < cols {
            col_split -= 1;
            trailing *= space.extent_at(col_split);
        }
        if trailing != cols {
            return Err(NoMatrix::new(space, rows, cols));
        }
        let mut row_split = col_split;
        let mut middle = 1;
        while row_split > 0 && middle < rows {
            row_split -= 1;
            middle *= space.extent_at(row_split);
        }
        if middle != rows {
            return Err(NoMatrix::new(space, rows, cols));
        }
        // Absorb degenerate leading axes so the answer is unique.
        while row_split > 0 && space.extent_at(row_split - 1) == 1 {
            row_split -= 1;
        }
        Ok(MatrixAxes {
            row_split,
            col_split,
        })
    }

    /// [`new`](Self::new) over a tile's whole box, the innermost (vectorized) axis in the columns.
    pub(crate) fn whole(space: &Space, rows: usize, cols: usize, vector_size: usize) -> Self {
        let rank = space.rank();
        let axes = MatrixAxes::new(space, rows, cols).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            axes.row_split == 0,
            "MatrixAxes::whole: this tile has axes above its {rows} rows, so its box is a \
             batch of matrices rather than one"
        );
        // Column edge and innermost extent are counted in lines, so no partial line is allowed.
        assert!(
            axes.col_split < rank,
            "MatrixAxes::whole: the innermost (vectorized) axis must be part of the column group"
        );
        let innermost = space.extent_at(rank - 1);
        assert!(
            innermost.is_multiple_of(vector_size),
            "MatrixAxes::whole: the innermost extent {innermost} is not a whole number of \
             {vector_size}-wide lines"
        );
        axes
    }
}

/// No grouping of a space's axes multiplies out to `rows x cols`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct NoMatrix {
    rows: usize,
    cols: usize,
    extents: Vec<usize>,
}

impl NoMatrix {
    fn new(space: &Space, rows: usize, cols: usize) -> Self {
        NoMatrix {
            rows,
            cols,
            extents: (0..space.rank()).map(|p| space.extent_at(p)).collect(),
        }
    }
}

impl Display for NoMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatrixAxes: no grouping of this tile's axes gives a {}x{} matrix (its extents are \
             {:?})",
            self.rows, self.cols, self.extents
        )
    }
}

impl std::error::Error for NoMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axis;

    const OH: Axis = Axis(0);
    const RH: Axis = Axis(1);
    const CI: Axis = Axis(2);

    /// A run along the columns takes the innermost axes whole, outwards, until it is covered.
    #[test]
    fn a_col_run_takes_the_innermost_axes_whole() {
        const KB: Axis = Axis(3);
        let blocked = Space::new(&[(OH, 8), (KB, 4), (CI, 16)]);
        let axes = MatrixAxes::new(&blocked, 8, 64).unwrap();
        assert_eq!(axes.col_run(&blocked, 32), vec![1, 2, 16]);
        assert_eq!(axes.col_run(&blocked, 8), vec![1, 1, 8]);
        let whole = Space::new(&[(OH, 8), (CI, 64)]);
        assert_eq!(
            MatrixAxes::trailing(&whole).col_run(&whole, 32),
            vec![1, 32]
        );
    }

    /// One axis per edge.
    #[test]
    fn a_rank_two_operand_splits_between_its_axes() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 4, 1).col_split, 1);
    }

    /// A two-axis contraction leaves only the output axis in the row group.
    #[test]
    fn a_contraction_over_two_axes_splits_before_both() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 8, 8, 1).col_split, 1);
    }

    #[test]
    fn both_edges_may_span_several_axes() {
        let s = Space::new(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(MatrixAxes::whole(&s, 12, 2, 2).col_split, 2);
    }

    /// The smallest column group wins, so a degenerate leading axis stays in the rows.
    #[test]
    fn a_degenerate_leading_axis_stays_in_the_row_group() {
        let s = Space::new(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixAxes::whole(&s, 1, 8, 1).col_split, 1);
    }

    /// 8x8 is not a face of the box.
    #[test]
    #[should_panic(expected = "no grouping of this tile's axes")]
    fn a_mismatched_fragment_is_refused() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 2)]);
        MatrixAxes::whole(&s, 8, 8, 1);
    }

    /// The column group must contain the innermost axis.
    #[test]
    #[should_panic(expected = "innermost (vectorized) axis")]
    fn the_vectorized_axis_must_land_in_the_column_group() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        MatrixAxes::whole(&s, 32, 1, 1);
    }

    /// The line width must divide the innermost extent.
    #[test]
    #[should_panic(expected = "whole number of 4-wide lines")]
    fn a_partial_innermost_line_is_refused() {
        let s = Space::new(&[(OH, 8), (CI, 6)]);
        MatrixAxes::whole(&s, 8, 6, 4);
    }
}
