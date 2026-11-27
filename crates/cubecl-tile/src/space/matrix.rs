//! Which of a space's dimensions form the matrix a 2-D reader sees.

use core::fmt::{self, Display, Formatter};

use crate::{Digit, Extent, Quotient, Radix, Space, TermStep, WalkUnits};

/// Which of a tile's dimensions form a 2-D matrix: a batch prefix, then the row group, then the
/// column group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatrixDimensions {
    /// Where the row group starts; everything before it is batch.
    pub row_split: usize,
    /// Where the column group starts.
    pub col_split: usize,
}

impl MatrixDimensions {
    /// The trailing pair: leading dimensions batch, the last two the matrix.
    pub(crate) fn trailing(space: &Space) -> Self {
        let rank = space.rank();
        MatrixDimensions {
            row_split: rank - 2,
            col_split: rank - 1,
        }
    }

    /// The innermost dimension as columns and the last dimension above it of extent past one as
    /// rows.
    pub(crate) fn edges(space: &Space) -> Self {
        let rank = space.rank();
        let col_split = rank - 1;
        // A dynamic dimension is not a number one.
        let row_split = (0..col_split)
            .rev()
            .find(|&p| !matches!(space.extent_raw(space.dimension_at(p)), Extent::Static(1)))
            .unwrap_or(col_split.saturating_sub(1));
        MatrixDimensions {
            row_split,
            col_split,
        }
    }

    /// An operand's matrix against the output `out` it contracts into: the dimensions `out` lacks
    /// as one edge, however many dimensions make up the contraction, and the free dimension beside
    /// them as the other. The contraction is the window's trailing run of dimensions `out` lacks,
    /// or the run below its free trailing dimension.
    pub(crate) fn operand(window: &Space, out: &Space) -> Self {
        let rank = window.rank();
        let contracted = |p: usize| !out.contains(window.dimension_at(p));
        // The run of contracted dimensions ending at `end`, exclusive: its first position.
        let run_start = |end: usize| {
            let mut start = end;
            while start > 0 && contracted(start - 1) {
                start -= 1;
            }
            start
        };
        match contracted(rank - 1) {
            // `{.., free, k..}`: the contraction the columns, the free dimension above it the rows.
            true => {
                let col_split = run_start(rank);
                MatrixDimensions {
                    row_split: col_split.saturating_sub(1),
                    col_split,
                }
            }
            // `{.., k.., free}`: the free dimension the columns, the contraction above it the rows.
            false => MatrixDimensions {
                row_split: run_start(rank - 1),
                col_split: rank - 1,
            },
        }
    }

    /// An accumulator's matrix, against the lhs it is contracted with.
    pub(crate) fn accumulator(acc: &Space, lhs: &Space) -> Self {
        let mut col_split = acc.rank() - 1;
        while col_split > 1 && !lhs.contains(acc.dimension_at(col_split - 1)) {
            col_split -= 1;
        }
        MatrixDimensions {
            row_split: col_split - 1,
            col_split,
        }
    }

    /// The position a matrix's rows step along: the innermost dimension of the row group longer
    /// than one, as a dimension of one in a window may carry any stride.
    pub(crate) fn row_dimension(&self, space: &Space) -> usize {
        (self.row_split..self.col_split)
            .rev()
            .find(|&p| !matches!(space.extent_raw(space.dimension_at(p)), Extent::Static(1)))
            .unwrap_or(self.col_split.saturating_sub(1))
    }

    /// The row edge these dimensions give in `space`.
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

    /// A run of `values` scalars along the columns from their origin, as an extent per dimension of
    /// `space`: the innermost dimensions taken whole, outwards, until the run is covered, every
    /// other dimension one. Over a `k` split in blocks of 16, a run of 32 is the whole of the
    /// position inside a block, two blocks along.
    ///
    /// # Panics
    ///
    /// Where the column dimensions cover no run of `values` whole.
    pub(crate) fn col_run(&self, space: &Space, values: usize) -> Vec<usize> {
        let mut run = vec![1; space.rank()];
        let mut left = values;
        for p in (self.col_split..space.rank()).rev() {
            let taken = space.extent_at(p).min(left);
            assert!(
                left.is_multiple_of(taken),
                "MatrixDimensions::col_run: a run of {values} values along the columns, which these \
                 dimensions do not cover whole"
            );
            run[p] = taken;
            left /= taken;
        }
        run
    }

    /// The walk that moving `coords_per_count` columns a count is over `space`, whose dimensions
    /// count `counts` coordinates each: the column index read as one digit per column dimension
    /// ([`Digit::step`]), each dimension it moves walked by its digit's step, those it holds and
    /// those one coordinate long left out. `None` where a column dimension would carry into the
    /// next, which no constant step per dimension is.
    pub(crate) fn col_walk(
        &self,
        space: &Space,
        counts: &[usize],
        coords_per_count: usize,
    ) -> Option<WalkUnits> {
        MatrixDimensions::group_walk(
            space,
            counts,
            self.col_split..space.rank(),
            coords_per_count,
        )
    }

    /// [`col_walk`](Self::col_walk) along the rows.
    pub(crate) fn row_walk(
        &self,
        space: &Space,
        counts: &[usize],
        coords_per_count: usize,
    ) -> Option<WalkUnits> {
        MatrixDimensions::group_walk(
            space,
            counts,
            self.row_split..self.col_split,
            coords_per_count,
        )
    }

    /// The walk that moving `coords_per_count` along the index `group` unravels over is.
    fn group_walk(
        space: &Space,
        counts: &[usize],
        group: core::ops::Range<usize>,
        coords_per_count: usize,
    ) -> Option<WalkUnits> {
        // A dimension one coordinate long never moves, so the first longer one is the coarsest.
        let moving: Vec<usize> = group.clone().filter(|&p| counts[p] > 1).collect();
        let mut walked = Vec::new();
        for (at, &p) in moving.iter().enumerate() {
            let digit = Digit {
                place_value: Radix::Static(counts[p + 1..group.end].iter().product()),
                extent: Radix::Static(counts[p]),
                quotient: match at {
                    0 => Quotient::Whole,
                    _ => Quotient::Modulo,
                },
            };
            match digit.step(coords_per_count) {
                TermStep::Adds(0) | TermStep::Holds => {}
                TermStep::Adds(step) => walked.push((space.dimension_at(p), step as usize)),
                TermStep::AddsCoefficient(_) | TermStep::Reevaluates => return None,
            }
        }
        Some(WalkUnits::over(&walked))
    }

    /// The dimensions giving a `rows x cols` matrix, both scalar, found from the innermost
    /// dimension outwards.
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
        // Absorb degenerate leading dimensions so the answer is unique.
        while row_split > 0 && space.extent_at(row_split - 1) == 1 {
            row_split -= 1;
        }
        Ok(MatrixDimensions {
            row_split,
            col_split,
        })
    }

    /// [`new`](Self::new) over a tile's whole box, the innermost (vectorized) dimension in the
    /// columns.
    pub(crate) fn whole(space: &Space, rows: usize, cols: usize, vector_size: usize) -> Self {
        let rank = space.rank();
        let dimensions = MatrixDimensions::new(space, rows, cols).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            dimensions.row_split == 0,
            "MatrixDimensions::whole: this tile has dimensions above its {rows} rows, so its box is a \
             batch of matrices rather than one"
        );
        // Column edge and innermost extent are counted in lines, so no partial line is allowed.
        assert!(
            dimensions.col_split < rank,
            "MatrixDimensions::whole: the innermost (vectorized) dimension must be part of the column group"
        );
        let innermost = space.extent_at(rank - 1);
        assert!(
            innermost.is_multiple_of(vector_size),
            "MatrixDimensions::whole: the innermost extent {innermost} is not a whole number of \
             {vector_size}-wide lines"
        );
        dimensions
    }
}

/// No grouping of a space's dimensions multiplies out to `rows x cols`.
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
            "MatrixDimensions: no grouping of this tile's dimensions gives a {}x{} matrix (its extents are \
             {:?})",
            self.rows, self.cols, self.extents
        )
    }
}

impl std::error::Error for NoMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dimension;

    const OH: Dimension = Dimension(0);
    const RH: Dimension = Dimension(1);
    const CI: Dimension = Dimension(2);

    /// Over `k` in blocks of 16 lines, a step of 32 lines moves the block by two and holds the
    /// position inside it; a step of 8 moves inside a block and would carry into the next one.
    #[test]
    fn a_column_walk_steps_each_dimension_of_the_group_or_carries() {
        const KB: Dimension = Dimension(3);
        let blocked = Space::new(&[(OH, 8), (KB, 4), (CI, 16)]);
        let dimensions = MatrixDimensions::new(&blocked, 8, 64).unwrap();
        let counts = [8, 4, 16];
        assert_eq!(
            dimensions.col_walk(&blocked, &counts, 32),
            Some(WalkUnits::along(KB, 2))
        );
        assert_eq!(dimensions.col_walk(&blocked, &counts, 8), None);
        let whole = Space::new(&[(OH, 8), (CI, 64)]);
        assert_eq!(
            MatrixDimensions::trailing(&whole).col_walk(&whole, &[8, 8], 2),
            Some(WalkUnits::along(CI, 2))
        );
        assert_eq!(
            MatrixDimensions::trailing(&whole).row_walk(&whole, &[8, 8], 3),
            Some(WalkUnits::along(OH, 3))
        );
    }

    /// A run along the columns takes the innermost dimensions whole, outwards, until it is covered.
    #[test]
    fn a_col_run_takes_the_innermost_dimensions_whole() {
        const KB: Dimension = Dimension(3);
        let blocked = Space::new(&[(OH, 8), (KB, 4), (CI, 16)]);
        let dimensions = MatrixDimensions::new(&blocked, 8, 64).unwrap();
        assert_eq!(dimensions.col_run(&blocked, 32), vec![1, 2, 16]);
        assert_eq!(dimensions.col_run(&blocked, 8), vec![1, 1, 8]);
        let whole = Space::new(&[(OH, 8), (CI, 64)]);
        assert_eq!(
            MatrixDimensions::trailing(&whole).col_run(&whole, 32),
            vec![1, 32]
        );
    }

    /// One dimension per edge.
    #[test]
    fn a_rank_two_operand_splits_between_its_dimensions() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        assert_eq!(MatrixDimensions::whole(&s, 8, 4, 1).col_split, 1);
    }

    /// A two-dimension contraction leaves only the output dimension in the row group.
    #[test]
    fn a_contraction_over_two_dimensions_splits_before_both() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixDimensions::whole(&s, 8, 8, 1).col_split, 1);
    }

    #[test]
    fn both_edges_may_span_several_dimensions() {
        let s = Space::new(&[(OH, 3), (RH, 4), (CI, 2)]);
        assert_eq!(MatrixDimensions::whole(&s, 12, 2, 2).col_split, 2);
    }

    /// The smallest column group wins, so a degenerate leading dimension stays in the rows.
    #[test]
    fn a_degenerate_leading_dimension_stays_in_the_row_group() {
        let s = Space::new(&[(OH, 1), (RH, 2), (CI, 4)]);
        assert_eq!(MatrixDimensions::whole(&s, 1, 8, 1).col_split, 1);
    }

    /// 8x8 is not a face of the box.
    #[test]
    #[should_panic(expected = "no grouping of this tile's dimensions")]
    fn a_mismatched_fragment_is_refused() {
        let s = Space::new(&[(OH, 8), (RH, 2), (CI, 2)]);
        MatrixDimensions::whole(&s, 8, 8, 1);
    }

    /// The column group must contain the innermost dimension.
    #[test]
    #[should_panic(expected = "innermost (vectorized) dimension")]
    fn the_vectorized_dimension_must_land_in_the_column_group() {
        let s = Space::new(&[(OH, 8), (CI, 4)]);
        MatrixDimensions::whole(&s, 32, 1, 1);
    }

    /// The line width must divide the innermost extent.
    #[test]
    #[should_panic(expected = "whole number of 4-wide lines")]
    fn a_partial_innermost_line_is_refused() {
        let s = Space::new(&[(OH, 8), (CI, 6)]);
        MatrixDimensions::whole(&s, 8, 6, 4);
    }
}
