//! The block and depth a contraction is cut to, shared by every nest schedule.

use crate::*;

#[derive(Clone, Debug)]
pub(crate) struct ContractShape {
    /// The accumulator's space.
    pub space: Space,
    /// The accumulator's matrix edges.
    pub acc_axes: MatrixAxes,
    /// The contracted axes.
    pub reduce: Vec<Axis>,
    /// Their extents, off the operands' merged space.
    pub reduce_extents: Vec<usize>,
    /// The contraction depth, every contracted axis multiplied out.
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
        let acc_axes = MatrixAxes::accumulator(&space, lhs);
        let reduce = Space::contracted(&[lhs, rhs], &space).to_vec();
        let reduce_extents = reduce
            .iter()
            .map(|&axis| merged.extent(axis))
            .collect::<Vec<_>>();
        let cols = acc_axes.cols(&space);
        let spread = if contracted_per_step > 1 { 1 } else { vw / aw };
        // A spread column rounds up: its last cell may be short.
        let cell = column_cell_width(contracted_per_step, spread, vw);
        let nr = if spread > 1 {
            cols.div_ceil(cell)
        } else {
            cols / cell
        };

        let mr = acc_axes.rows(&space);

        Self {
            kc: reduce_extents.iter().product(),
            mr,
            nr,
            cols,
            space,
            acc_axes,
            reduce,
            reduce_extents,
            contracted_per_step,
            spread,
            lw,
            vw,
            aw,
        }
    }

    /// The axes a 2-D reading takes both operands over, if one describes them.
    pub(crate) fn matrix_axes(&self, lhs: &Space, rhs: &Space) -> Option<(MatrixAxes, MatrixAxes)> {
        let lhs_axes = MatrixAxes::new(lhs, self.mr, self.kc).ok()?;
        let rhs_axes = match self.contracted_per_step > 1 {
            true => MatrixAxes::new(rhs, self.cols, self.kc).ok()?,
            false => MatrixAxes::new(rhs, self.kc, self.cols).ok()?,
        };
        Some((lhs_axes, rhs_axes))
    }

    /// The lhs's `mr x kc` matrix axes for the 2-D nest.
    pub(crate) fn lhs_axes(&self, lhs: &Space) -> MatrixAxes {
        MatrixAxes::new(lhs, self.mr, self.kc).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The rhs's matrix axes: `(col, k)` at a folded step, `(k, col)` otherwise.
    pub(crate) fn rhs_axes(&self, rhs: &Space) -> MatrixAxes {
        match self.contracted_per_step > 1 {
            true => MatrixAxes::new(rhs, self.cols, self.kc).unwrap_or_else(|e| panic!("{e}")),
            false => MatrixAxes::new(rhs, self.kc, self.cols).unwrap_or_else(|e| panic!("{e}")),
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
            self.acc_axes.row_split,
            self.acc_axes.col_split,
        )
    }

    /// The extents `col` unravels over, the innermost counted in block-column cells.
    pub(crate) fn column_line_extents(&self) -> Vec<usize> {
        line_extents(
            &self.space,
            self.cell_width(),
            self.acc_axes.col_split,
            self.space.rank(),
        )
    }

    /// The accumulator's batch extents.
    pub(crate) fn batch_extents(&self) -> Vec<usize> {
        (0..self.acc_axes.row_split)
            .map(|p| self.space.extent_at(p))
            .collect()
    }

    /// How many batch matrices a nest walks.
    pub(crate) fn matrices(&self) -> usize {
        self.batch_extents().iter().product()
    }

    /// The block's size in scalars, as [`RegisterBlock::budget`] counts it.
    pub(crate) fn scalars(&self) -> usize {
        self.mr * self.nr * self.contracted_per_step * self.aw * self.spread
    }

    /// Whether the unit fan-out's fixed extracts stay in step with the flat walk's coordinate.
    pub(crate) fn component_index_exact(&self) -> bool {
        self.reduce.len() == 1
            || self.reduce_extents[self.reduce_extents.len() - 1].is_multiple_of(self.lw)
    }
}

/// Scalars per block column: one at a folded step, `spread` when spread, else the rhs width.
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

    const B: Axis = Axis(0);
    const OH: Axis = Axis(1);
    const OW: Axis = Axis(2);
    const C: Axis = Axis(3);
    const RH: Axis = Axis(4);
    const RW: Axis = Axis(5);

    /// A depthwise-shaped cell coordinate covers every accumulator axis.
    #[test]
    fn the_cell_coordinate_covers_every_accumulator_axis() {
        let acc = Space::new(&[(B, 1), (OH, 1), (OW, 4), (C, 4)]);
        let lhs = Space::new(&[(C, 4), (RH, 3), (RW, 3)]);
        let rhs = Space::new(&[(B, 1), (OH, 1), (OW, 4), (C, 4), (RH, 3), (RW, 3)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.acc_axes.col_split, 1);
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

    /// One axis per edge gives identity unravels.
    #[test]
    fn a_single_axis_per_edge_leaves_the_coordinate_unchanged() {
        let acc = Space::new(&[(B, 2), (OH, 4), (C, 8)]);
        let lhs = Space::new(&[(B, 2), (OH, 4), (RH, 6)]);
        let rhs = Space::new(&[(B, 2), (RH, 6), (C, 8)]);

        let shape = ContractShape::new(&lhs, &rhs, acc.clone(), 1, 4, 4, 4);

        assert_eq!(shape.batch_extents(), vec![2]);
        assert_eq!(shape.row_extents(), vec![4]);
        assert_eq!(shape.column_line_extents(), vec![2]);
    }
}
