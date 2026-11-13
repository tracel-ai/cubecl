//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

use super::base::{GatherProblem, LhsVectorAxis};
use crate::ops::contraction::{AccCoords, resolve_nd_coords};
use crate::ops::matmul::leaf::Side;

/// `side` read at accumulator cell `(row, col)` of batch matrix `batch`.
#[cube]
pub(super) fn cell_read<T: Numeric, W: Size>(
    view: &Masked<'_, Vector<T, W>, CoordsDyn>,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] side: Side,
    #[comptime] problem: GatherProblem,
) -> Vector<T, W> {
    view.read(cell_position(batch, row, col, reduce_coords, side, problem))
}

/// Resolve one side's N-D coordinate from an accumulator cell and contracted coordinates. A
/// row-vectorized lhs counts its rows in cells, since each component is a row of its own.
#[cube]
pub(super) fn cell_position(
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] side: Side,
    #[comptime] problem: GatherProblem,
) -> CoordsDyn {
    let acc_coords = acc_cell_coords(
        batch,
        row,
        col,
        problem.block.row_extents(),
        problem.block.column_line_extents(),
    );
    let acc_counts = comptime!(match (side, problem.lhs_vector_axis) {
        (Side::Lhs, LhsVectorAxis::Rows) => AccCoords::Cells,
        _ => AccCoords::Lines,
    });
    resolve_nd_coords(
        problem.space(side),
        &problem.block.space,
        &problem.block.reduce,
        &acc_coords,
        reduce_coords,
        problem.width(side),
        acc_counts,
    )
}

/// `coords` with its last entry offset by `delta`.
#[cube]
pub(super) fn offset_last(coords: &CoordsDyn, #[comptime] rank: usize, delta: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        out.push(if p == rank - 1 {
            coords[p].plus(delta)
        } else {
            coords[p]
        });
    }
    out
}

/// The accumulator cell coordinate [`resolve_nd_coords`] reads, one entry per axis of the
/// accumulator's space in its own order.
#[cube]
fn acc_cell_coords(
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] row_extents: Vec<usize>,
    #[comptime] col_extents: Vec<usize>,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for p in 0..batch.len() {
        out.push(batch.at(p));
    }
    let rows = Coords::constant(row_extents.clone()).unravel(row);
    #[unroll]
    for p in 0..rows.len() {
        out.push(rows.at(p));
    }
    let cols = Coords::constant(col_extents.clone()).unravel(col);
    #[unroll]
    for p in 0..cols.len() {
        out.push(cols.at(p));
    }

    out
}

/// Contracted coordinates of tap `tap_coords` with the fastest axis at `last`.
#[cube]
pub(super) fn step_coords(tap_coords: &Coords<u32>, last: u32) -> Coords<u32> {
    let mut out = Coords::<u32>::new();
    out.extend(tap_coords);
    out.push(last);
    out
}
