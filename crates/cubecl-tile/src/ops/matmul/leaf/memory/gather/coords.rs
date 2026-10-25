//! Coordinate resolution shared by the general and separable gather schedules.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

use super::base::GatherProblem;
use crate::ops::contraction::resolve_nd_coords;

/// One operand read at accumulator cell `(row, col)` of batch matrix `batch`.
#[cube]
pub(super) fn cell_read<T: Numeric, W: Size>(
    view: &Masked<'_, Vector<T, W>, CoordsDyn>,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] operand: Space,
    #[comptime] problem: GatherProblem,
    #[comptime] width: usize,
) -> Vector<T, W> {
    view.read(cell_position(
        batch,
        row,
        col,
        reduce_coords,
        operand,
        problem,
        width,
    ))
}

/// Resolve one operand's N-D coordinate from an accumulator cell and contracted coordinates.
#[cube]
pub(super) fn cell_position(
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    reduce_coords: &Coords<u32>,
    #[comptime] operand: Space,
    #[comptime] problem: GatherProblem,
    #[comptime] width: usize,
) -> CoordsDyn {
    let acc_coords = acc_cell_coords(
        batch,
        row,
        col,
        comptime!(problem.block.row_extents()),
        comptime!(problem.block.column_line_extents()),
    );
    resolve_nd_coords(
        operand,
        comptime!(problem.block.space.clone()),
        comptime!(problem.block.reduce.clone()),
        &acc_coords,
        reduce_coords,
        width,
        false,
    )
}

/// `coords` with its last entry offset by `delta`.
#[cube]
pub(super) fn offset_last(coords: &CoordsDyn, #[comptime] rank: usize, delta: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        out.push(if comptime!(p == rank - 1) {
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
    let rows = Coords::constant(comptime!(row_extents.clone())).unravel(row);
    #[unroll]
    for p in 0..rows.len() {
        out.push(rows.at(p));
    }
    let cols = Coords::constant(comptime!(col_extents.clone())).unravel(col);
    #[unroll]
    for p in 0..cols.len() {
        out.push(cols.at(p));
    }

    out
}
