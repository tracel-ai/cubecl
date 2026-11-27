//! The block and depth a contraction is cut to, shared by every nest schedule.

use crate::*;

#[derive(Clone, Debug)]
pub(crate) struct ContractShape {
    /// The accumulator's space.
    pub space: Space,
    /// The accumulator's matrix edges.
    pub acc_dimensions: MatrixDimensions,
    /// The contracted dimensions.
    pub reduce: Vec<Dimension>,
    /// Their extents, off the operands' merged space.
    pub reduce_extents: Vec<usize>,
    /// The contraction depth, every contracted dimension multiplied out.
    pub kc: usize,
    /// The block's rows.
    pub mr: usize,
    /// Its columns, counted in cells.
    pub nr: usize,
    /// The accumulator's column extent in scalars.
    pub cols: usize,
    /// Contracted values one step consumes.
    pub contracted_per_step: usize,
    /// How many sink cells one block column's vector components spread across.
    pub spread: usize,
    /// The lhs's line width.
    pub lw: usize,
    /// The rhs's, and so the block's.
    pub vw: usize,
    /// The accumulator's.
    pub aw: usize,
}

impl ContractShape {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        lhs: &Space,
        rhs: &Space,
        space: Space,
        contracted_per_step: usize,
        lw: usize,
        vw: usize,
        aw: usize,
    ) -> Self {
        let merged = Space::merge(&[lhs, rhs]);
        let acc_dimensions = MatrixDimensions::accumulator(&space, lhs);
        let reduce = Space::contracted(&[lhs, rhs], &space).to_vec();
        let reduce_extents = reduce
            .iter()
            .map(|&dimension| merged.extent(dimension))
            .collect::<Vec<_>>();
        let cols = acc_dimensions.cols(&space);
        let spread = if contracted_per_step > 1 { 1 } else { vw / aw };
        // A spread column rounds up: its last cell may be short.
        let cell = column_cell_width(contracted_per_step, spread, vw);
        let nr = if spread > 1 {
            cols.div_ceil(cell)
        } else {
            cols / cell
        };

        let mr = acc_dimensions.rows(&space);

        Self {
            kc: reduce_extents.iter().product(),
            mr,
            nr,
            cols,
            space,
            acc_dimensions,
            reduce,
            reduce_extents,
            contracted_per_step,
            spread,
            lw,
            vw,
            aw,
        }
    }

    /// The dimensions a 2-D reading takes both operands over, if one describes them.
    pub(crate) fn matrix_dimensions(
        &self,
        lhs: &Space,
        rhs: &Space,
    ) -> Option<(MatrixDimensions, MatrixDimensions)> {
        let lhs_dimensions = MatrixDimensions::new(lhs, self.mr, self.kc).ok()?;
        let rhs_dimensions = match self.contracted_per_step > 1 {
            true => MatrixDimensions::new(rhs, self.cols, self.kc).ok()?,
            false => MatrixDimensions::new(rhs, self.kc, self.cols).ok()?,
        };
        Some((lhs_dimensions, rhs_dimensions))
    }

    /// The lhs's `mr x kc` matrix dimensions for the 2-D nest.
    pub(crate) fn lhs_dimensions(&self, lhs: &Space) -> MatrixDimensions {
        MatrixDimensions::new(lhs, self.mr, self.kc).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The rhs's matrix dimensions: `(col, k)` at a summing step, `(k, col)` otherwise.
    pub(crate) fn rhs_dimensions(&self, rhs: &Space) -> MatrixDimensions {
        match self.contracted_per_step > 1 {
            true => {
                MatrixDimensions::new(rhs, self.cols, self.kc).unwrap_or_else(|e| panic!("{e}"))
            }
            false => {
                MatrixDimensions::new(rhs, self.kc, self.cols).unwrap_or_else(|e| panic!("{e}"))
            }
        }
    }

    /// How many of the accumulator's innermost scalars one block column holds.
    pub(crate) fn cell_width(&self) -> usize {
        column_cell_width(self.contracted_per_step, self.spread, self.vw)
    }

    /// The extents `row` unravels over.
    pub(crate) fn row_extents(&self) -> Vec<usize> {
        line_extents(
            &self.space,
            1,
            self.acc_dimensions.row_split,
            self.acc_dimensions.col_split,
        )
    }

    /// The extents `col` unravels over, the innermost counted in block-column cells.
    pub(crate) fn column_line_extents(&self) -> Vec<usize> {
        line_extents(
            &self.space,
            self.cell_width(),
            self.acc_dimensions.col_split,
            self.space.rank(),
        )
    }

    /// The accumulator's batch extents.
    pub(crate) fn batch_extents(&self) -> Vec<usize> {
        (0..self.acc_dimensions.row_split)
            .map(|p| self.space.extent_at(p))
            .collect()
    }

    /// How many batch matrices a nest walks.
    pub(crate) fn matrices(&self) -> usize {
        self.batch_extents().iter().product()
    }

    /// The block's size in scalars, as [`SoftwareMatmul::budget`] counts it.
    pub(crate) fn scalars(&self) -> usize {
        self.mr * self.nr * self.contracted_per_step * self.aw * self.spread
    }

    /// The fastest contracted dimension, the one a walk steps innermost.
    pub(crate) fn fastest_dimension(&self) -> Dimension {
        self.reduce[self.reduce.len() - 1]
    }

    /// The walk over every contracted dimension, a coordinate a count: how a cell's contracted
    /// positions are reached from where they are all zero.
    pub(crate) fn reduce_walk(&self) -> WalkUnits {
        WalkUnits::over(&self.reduce.iter().map(|&d| (d, 1)).collect::<Vec<_>>())
    }

    /// The walk along the fastest contracted dimension alone, a coordinate a count.
    pub(crate) fn fastest_walk(&self) -> WalkUnits {
        WalkUnits::along(self.fastest_dimension(), 1)
    }

    /// The fastest contracted dimension's extent, in scalars.
    pub(crate) fn fastest_extent(&self) -> usize {
        self.reduce_extents[self.reduce_extents.len() - 1]
    }

    /// Whether the unit fan-out's fixed extracts stay in step with the flat walk's coordinate.
    pub(crate) fn component_index_exact(&self) -> bool {
        self.reduce.len() == 1 || self.fastest_extent().is_multiple_of(self.lw)
    }
}

/// Scalars per block column: one at a summing step, `spread` when spread, else the rhs width.
fn column_cell_width(contracted_per_step: usize, spread: usize, vw: usize) -> usize {
    if contracted_per_step > 1 {
        1
    } else if spread > 1 {
        spread
    } else {
        vw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Dimension = Dimension(0);
    const OH: Dimension = Dimension(1);
    const OW: Dimension = Dimension(2);
    const C: Dimension = Dimension(3);
    const RH: Dimension = Dimension(4);
    const RW: Dimension = Dimension(5);

    /// A depthwise-shaped cell coordinate covers every accumulator dimension.
    #[test]
    fn the_cell_coordinate_covers_every_accumulator_dimension() {
        let acc = Space::new(&[(B, 1), (OH, 1), (OW, 4), (C, 4)]);
        let lhs = Space::new(&[(C, 4), (RH, 3), (RW, 3)]);
        let rhs = Space::new(&[(B, 1), (OH, 1), (OW, 4), (C, 4), (RH, 3), (RW, 3)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.acc_dimensions.col_split, 1);
        assert_eq!(
            shape.batch_extents().len()
                + shape.row_extents().len()
                + shape.column_line_extents().len(),
            acc.rank()
        );
        assert_eq!(shape.row_extents().iter().product::<usize>(), shape.mr);
        assert_eq!(
            shape.column_line_extents().iter().product::<usize>(),
            shape.nr
        );
    }

    /// One dimension per edge gives identity unravels.
    #[test]
    fn a_single_dimension_per_edge_leaves_the_coordinate_unchanged() {
        let acc = Space::new(&[(B, 2), (OH, 4), (C, 8)]);
        let lhs = Space::new(&[(B, 2), (OH, 4), (RH, 6)]);
        let rhs = Space::new(&[(B, 2), (RH, 6), (C, 8)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.batch_extents(), vec![2]);
        assert_eq!(shape.row_extents(), vec![4]);
        assert_eq!(shape.column_line_extents(), vec![2]);
    }
}
