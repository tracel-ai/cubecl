//! How a plane's units reach the cells of a sum one at a time: in turns over a plane's cells
//! ([`in_plane_turns`]), and written through a memory window's own write ([`write_cells`]).

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// Hands `work` each cell of `cells` this unit takes when a plane's units take them in turns, a
/// cell each per turn: this unit's place in the plane, then every plane width past it.
///
/// # Guarantees
///
/// - Across a plane's units, every cell below `cells` is handed to exactly one unit, once, for
///   any plane width.
pub fn in_plane_turns(_cells: usize, _work: impl FnMut(usize)) {
    unexpanded!()
}

/// [`in_plane_turns`] as a kernel expands it.
pub mod in_plane_turns {
    use super::*;

    pub fn expand(
        scope: &Scope,
        cells: NativeExpand<usize>,
        work: impl FnMut(&Scope, NativeExpand<usize>),
    ) {
        let (first, stride) = first_cell_and_stride::expand(scope);
        let turns = range_stepped::expand(scope, first, cells, stride);
        for_expand(scope, turns, false, work);
    }
}

/// This unit's first cell of a plane's turns, and the cells between two of its turns.
#[cube]
fn first_cell_and_stride() -> (usize, usize) {
    (UNIT_POS_PLANE as usize, PLANE_DIM as usize)
}

/// Writes a sum's cells into `mem` through its own write, which replaces them or adds into them:
/// `cells` writes each run of `W` cells it holds into the sink it is handed, at the run's row and
/// line of the accumulator's trailing dimensions of `space`.
///
/// # Guarantees
///
/// - The sink addresses `mem`'s window as the accumulator's rows and columns, whatever order
///   `space` lists the window's dimensions in past its trailing ones.
pub fn write_cells<Out: Numeric, W: Size>(
    _mem: &mut Memory,
    _space: &Space,
    _cells: impl FnOnce(&mut MatrixViewMut<'_, Vector<Out, W>>),
) {
    unexpanded!()
}

/// [`write_cells`] as a kernel expands it.
pub mod write_cells {
    use super::*;

    pub fn expand<'a, Out: Numeric, W: Size>(
        scope: &Scope,
        mem: &'a mut MemoryExpand,
        space: &'a Space,
        cells: impl FnOnce(&Scope, &mut <MatrixViewMut<'a, Vector<Out, W>> as CubeType>::ExpandType),
    ) {
        let dimensions = MatrixDimensions::trailing(space);
        let mut sink =
            mem.__expand_matrix_mut_method::<Out, W>(scope, 0usize.into(), dimensions, space);
        cells(scope, &mut sink);
    }
}
