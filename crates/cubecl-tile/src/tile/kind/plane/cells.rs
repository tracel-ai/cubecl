//! How a plane's units reach the cells of a sum one at a time: in turns over a plane's cells
//! ([`plane_turns`]), and written through a memory window's own write ([`cell_sink`]).

use cubecl::prelude::*;

use crate::*;

/// This unit's first cell, and the cells between two of its turns, when a plane's units take a
/// sum's cells in turns, a cell each per turn: walked as `range_stepped(first, cells, stride)`.
///
/// # Guarantees
///
/// - Across a plane's units, that walk hands every cell below `cells` to exactly one unit, once,
///   for any plane width.
#[cube]
pub fn plane_turns() -> (usize, usize) {
    (UNIT_POS_PLANE as usize, PLANE_DIM as usize)
}

/// The sink a sum's cells are written into `mem` through, by its own write, which replaces them
/// or adds into them: each run of `W` cells at its row and line of the matrix `space`'s edges
/// make, its innermost dimension the columns and the last one above it longer than one cell the
/// rows, as the fragment's store intrinsic addresses them.
///
/// # Guarantees
///
/// - The sink addresses `mem`'s window as the accumulator's rows and columns, whatever order
///   `space` lists the window's dimensions in past its trailing ones, and whatever one-wide
///   dimensions it lists between them.
#[cube]
pub fn cell_sink<'a, Out: Numeric, W: Size>(
    mem: &'a mut Memory,
    #[comptime] space: &'a Space,
) -> MatrixViewMut<'a, Vector<Out, W>> {
    mem.matrix_mut::<Out, W>(0usize, comptime!(MatrixDimensions::edges(space)), space)
}
