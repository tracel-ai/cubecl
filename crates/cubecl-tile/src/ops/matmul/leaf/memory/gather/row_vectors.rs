//! The lhs vectors one contracted step reads, anchored once per tap where they advance by an add.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

use super::base::{GatherProblem, LhsRole};
use super::coords::{cell_position, step_coords};
use crate::ops::matmul::leaf::Side;

/// How a step's row vectors are read, decided once from the problem and the lhs's projection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowReads {
    /// The lhs spans the columns, so it is read per cell and no row vector is held.
    PerCell,
    /// Read through the masked view at every step.
    Checked,
    /// Anchored once per tap, then advanced by an add where the run is in bounds.
    Anchored,
}

/// The lhs vectors one contracted step reads for an lhs free of the columns.
#[derive(CubeType)]
pub(super) struct RowVectors<'a, EL: Numeric, L: Size> {
    /// Split only on the anchored path: a split refuses a backward window, which never anchors.
    lhs: &'a Tile,
    vectors: Array<Vector<EL, L>>,
    /// Each vector's physical anchor, `rank` entries a vector.
    anchors: Array<u32>,
    /// Whether each vector's run in the current tap is in bounds, as `1` or `0`.
    run_in_bounds: Array<u32>,
    #[cube(comptime)]
    reads: RowReads,
}

#[cube]
impl<'a, EL: Numeric, L: Size> RowVectors<'a, EL, L> {
    pub fn new(lhs: &'a Tile, #[comptime] problem: &GatherProblem) -> RowVectors<'a, EL, L> {
        let advanceable = lhs.run_advanceable(problem.block.fastest_axis());
        let reads = comptime!(match (problem.lhs, advanceable) {
            (LhsRole::PerCell, _) => RowReads::PerCell,
            (LhsRole::FreeOfColumn, false) => RowReads::Checked,
            (LhsRole::FreeOfColumn, true) => RowReads::Anchored,
        });
        let anchored = comptime!(reads == RowReads::Anchored);
        let projection = lhs.projection();
        let rank = comptime!(projection.physical_rank());
        let vectors = problem.row_vector_count();
        RowVectors::<'a, EL, L> {
            lhs,
            vectors: Array::new(vectors),
            anchors: Array::new(comptime!(if anchored { vectors * rank } else { 1 })),
            run_in_bounds: Array::new(comptime!(if anchored { vectors } else { 1 })),
            reads,
        }
    }

    /// Vector `row_vector` as the last [`read`](RowVectors::read) left it.
    pub fn vector(&self, row_vector: usize) -> Vector<EL, L> {
        self.vectors[row_vector]
    }

    /// Anchor each vector of the tap `tap_coords` at its first contracted step, recording whether
    /// its whole run lands in bounds, which holds when both ends do.
    pub fn anchor(
        &mut self,
        lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
        batch: &Coords<u32>,
        tap_coords: &Coords<u32>,
        #[comptime] problem: GatherProblem,
        #[comptime] unroll: bool,
    ) {
        if comptime!(self.reads == RowReads::Anchored) {
            let split = self.lhs.nd_split::<EL, L>();
            let rank = comptime!(split.rank);
            let rows_per_vector = problem.rows_per_vector();
            let fastest = problem.block.fastest_axis();
            let run = problem.block.fastest_extent();
            let first_step = step_coords(tap_coords, 0u32);
            let last_step = step_coords(tap_coords, comptime!((run - 1) as u32));
            #[unroll(unroll)]
            for row_vector in 0..problem.row_vector_count() {
                let row = (row_vector * rows_per_vector) as u32;
                let first = cell_position(
                    batch,
                    row,
                    0u32,
                    &first_step,
                    comptime!(Side::Lhs),
                    problem.clone(),
                );
                let last = cell_position(
                    batch,
                    row,
                    0u32,
                    &last_step,
                    comptime!(Side::Lhs),
                    problem.clone(),
                );
                let whole = lhs_view.is_in_bounds(first.clone()) && lhs_view.is_in_bounds(last);
                self.run_in_bounds[row_vector] = if whole { 1u32 } else { 0u32 };
                let anchor = split.map.anchor(first, comptime!(vec![fastest]));
                #[unroll]
                for pa in 0..rank {
                    self.anchors[row_vector * comptime!(rank) + pa] = anchor[pa];
                }
            }
        }
    }

    /// Read the vectors of the step at `reduce_coords`: advanced from the anchor where the run is
    /// in bounds, through the masked view otherwise.
    pub fn read(
        &mut self,
        lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
        batch: &Coords<u32>,
        reduce_coords: &Coords<u32>,
        #[comptime] problem: GatherProblem,
        #[comptime] unroll: bool,
    ) {
        let rows_per_vector = problem.rows_per_vector();
        match comptime!(self.reads) {
            RowReads::PerCell => {}
            RowReads::Checked =>
            {
                #[unroll(unroll)]
                for row_vector in 0..problem.row_vector_count() {
                    let row = (row_vector * rows_per_vector) as u32;
                    let pos = cell_position(
                        batch,
                        row,
                        0u32,
                        reduce_coords,
                        comptime!(Side::Lhs),
                        problem.clone(),
                    );
                    self.vectors[row_vector] = lhs_view.read(pos);
                }
            }
            RowReads::Anchored =>
            {
                #[unroll(unroll)]
                for row_vector in 0..problem.row_vector_count() {
                    let row = (row_vector * rows_per_vector) as u32;
                    let pos = cell_position(
                        batch,
                        row,
                        0u32,
                        reduce_coords,
                        comptime!(Side::Lhs),
                        problem.clone(),
                    );
                    self.vectors[row_vector] = if self.run_in_bounds[row_vector] == 1u32 {
                        self.advanced(row_vector, pos.clone(), problem.clone())
                    } else {
                        lhs_view.read(pos)
                    };
                }
            }
        }
    }

    /// The vector at `pos`, advanced from vector `row_vector`'s anchor and read unchecked.
    fn advanced(
        &self,
        row_vector: usize,
        pos: CoordsDyn,
        #[comptime] problem: GatherProblem,
    ) -> Vector<EL, L> {
        let split = self.lhs.nd_split::<EL, L>();
        let rank = comptime!(split.rank);
        let mut anchor = CoordsDyn::new();
        #[unroll]
        for pa in 0..rank {
            anchor.push(self.anchors[row_vector * comptime!(rank) + pa]);
        }
        let fastest = problem.block.fastest_axis();
        let at = split.map.advance(&anchor, pos, comptime!(vec![fastest]));
        split.view.read_unchecked(at)
    }
}
