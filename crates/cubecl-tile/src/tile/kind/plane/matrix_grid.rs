//! The `rows × cols` grid of fragments a plane partition's walked levels cut it into
//! ([`MatrixGrid`]).

use crate::*;

/// The `rows × cols` grid of fragments the walked `levels` cut `space` into.
pub(crate) fn partition_shape(space: &Space, levels: &[Level]) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut space = space.clone();
    for level in levels {
        let grid = MatrixGrid::new(level, &space);
        shape = (shape.0 * grid.rows, shape.1 * grid.cols);
        space = level.child(&space);
    }
    shape
}

/// The `rows × cols` grid of fragments a walked level cuts a partition into.
pub(crate) struct MatrixGrid {
    rows: usize,
    cols: usize,
}

impl MatrixGrid {
    /// The grid `level` cuts `space` into; a distributed level cuts nothing.
    pub(crate) fn new(level: &Level, space: &Space) -> Self {
        if level.coverage() != Coverage::Walk {
            return MatrixGrid { rows: 1, cols: 1 };
        }
        let edges = MatrixDimensions::edges(space);
        for (p, dimension) in space.dimensions().enumerate() {
            let tiles = level
                .tiles_const(space, dimension)
                .expect("plane partition level: tile counts must be comptime");
            assert!(
                p == edges.row_split || p == edges.col_split || tiles == 1,
                "plane partition level: leading (batch) dimensions must hand out one tile"
            );
        }
        MatrixGrid {
            rows: level
                .tiles_const(space, space.dimension_at(edges.row_split))
                .unwrap(),
            cols: level
                .tiles_const(space, space.dimension_at(edges.col_split))
                .unwrap(),
        }
    }

    /// Whether the level cuts the partition into more than one fragment.
    pub(crate) fn cuts(&self) -> bool {
        (self.rows, self.cols) != (1, 1)
    }
}
