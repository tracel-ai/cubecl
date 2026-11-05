//! The 2-D matrix views over a [`Tile`]: [`TileMatrix`] and its masked [`MatrixView`].

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coords2d, CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// A masked 2-D view: one matrix of a [`Tile`].
pub(crate) type MatrixView<'a, T> = Masked<'a, T, Coords2d>;
/// The mutable twin of [`MatrixView`].
pub(crate) type MatrixViewMut<'a, T> = MaskedMut<'a, T, Coords2d>;

/// A [`Layout`] presenting a tile's logical box as one `(row, col)` matrix: a pinned batch
/// prefix, then the axes `row` unravels over, then those `col` unravels over (in lines).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct TileMatrix {
    /// Leading coordinates, already resolved to this matrix.
    batches: Coords<u32>,
    /// The group `row` unravels over, in the space's axis order.
    row_extents: Coords<u32>,
    /// The group `col` unravels over, the innermost a line count.
    col_extents: Coords<u32>,
    tile_shape: Coords2d,
}

/// A [`TileMatrix`] over the operand's mapping.
pub(crate) type ProjectedMatrix = Projected<TileMatrix>;

#[cube]
impl TileMatrix {
    pub(crate) fn new(
        batches: Coords<u32>,
        row_extents: Coords<u32>,
        col_extents: Coords<u32>,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> Self {
        TileMatrix {
            batches,
            row_extents,
            col_extents,
            tile_shape: (rows as u32, cols as u32).runtime(),
        }
    }

    /// The `i`-th matrix of a tile whose window is `bound`, over the axes [`MatrixAxes`] names,
    /// each axis counted in `load`s: a run along the innermost axis for a plain buffer, and for one
    /// stored in tiles, every axis the load spans ([`VectorTile::counts`]).
    pub(crate) fn batch(
        bound: &Coords<u32>,
        #[comptime] space: &Space,
        #[comptime] composition: Composition,
        #[comptime] load: &VectorTile,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> Self {
        let rank = comptime!(space.rank());
        // Rounded up like the buffer's own load count, so a checked read covers a partial last
        // load.
        let counts = comptime!(load.counts(space));
        // A row, or a column, steps the innermost axis of its group, so a load may reach along that
        // axis and no other: the loads of a group are then consecutive rows (or columns), in order.
        // An axis of one inside a group steps nothing, so the innermost is the last one past one.
        let row_edge = comptime!(TileMatrix::innermost_past_one(
            space,
            axes.row_split..axes.col_split
        ));
        let col_edge = comptime!(TileMatrix::innermost_past_one(space, axes.col_split..rank));
        comptime!(assert!(
            (0..rank).all(|p| counts[p] == space.extent_at(p)
                || Some(p) == row_edge
                || Some(p) == col_edge),
            "TileMatrix::batch: a load of {:?} over {space:?} reaches along an axis that is neither the \
             innermost of the matrix's rows nor of its columns",
            load.extents()
        ));
        let rows = comptime!(
            counts[axes.row_split..axes.col_split]
                .iter()
                .product::<usize>()
        );
        let cols = comptime!(counts[axes.col_split..rank].iter().product::<usize>());
        let extents = Self::leading_extents(
            bound,
            comptime!(space),
            composition,
            comptime!(axes.row_split),
        );

        Self::new(
            extents.unravel(i.cast::<u32>()),
            Coords::constant(comptime!(counts[axes.row_split..axes.col_split].to_vec())),
            Coords::constant(comptime!(counts[axes.col_split..rank].to_vec())),
            rows,
            cols,
        )
    }

    /// The logical coordinate, in scalars, of the first value of line `(row, col)` of matrix `i`.
    pub(crate) fn value_coords(
        row: u32,
        col: u32,
        i: usize,
        #[comptime] space: &Space,
        #[comptime] axes: MatrixAxes,
        #[comptime] vector_size: usize,
    ) -> Coords<u32> {
        let rank = comptime!(space.rank());
        let mut coords = Coords::<u32>::new();
        let batches = Coords::constant(comptime!(
            (0..axes.row_split)
                .map(|p| space.extent_at(p))
                .collect::<Vec<_>>()
        ))
        .unravel(i.cast::<u32>());
        #[unroll]
        for p in 0..batches.len() {
            coords.push(batches.at(p));
        }
        let rows = Coords::constant(comptime!(
            (axes.row_split..axes.col_split)
                .map(|p| space.extent_at(p))
                .collect::<Vec<_>>()
        ))
        .unravel(row);
        #[unroll]
        for p in 0..rows.len() {
            coords.push(rows.at(p));
        }
        // The innermost column coordinate is a line index.
        let cols = Coords::constant(comptime!(line_extents(
            space,
            vector_size,
            axes.col_split,
            rank
        )))
        .unravel(col);
        let n = cols.len();
        #[unroll]
        for p in 0..n {
            if comptime!(p == n - 1) {
                coords.push(cols.at(p).times(comptime!(vector_size as u32)));
            } else {
                coords.push(cols.at(p));
            }
        }
        coords
    }

    /// The tile's whole logical box as one `rows x cols` matrix; `cols` is in scalars.
    pub(crate) fn whole(
        #[comptime] space: &Space,
        #[comptime] vector_size: usize,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> Self {
        let rank = comptime!(space.rank());
        let split = comptime!(MatrixAxes::whole(space, rows, cols, vector_size).col_split);

        Self::new(
            Coords::<u32>::new(),
            Coords::constant(comptime!(line_extents(space, vector_size, 0, split))),
            Coords::constant(comptime!(line_extents(space, vector_size, split, rank))),
            rows,
            comptime!(cols / vector_size),
        )
    }

    /// The leading (batch) extents a matrix index unravels over, in the space's axis order.
    fn leading_extents(
        bound: &Coords<u32>,
        #[comptime] space: &Space,
        #[comptime] composition: Composition,
        #[comptime] upto: usize,
    ) -> Coords<u32> {
        let mut out = Coords::<u32>::new();

        #[unroll]
        for p in 0..upto {
            // A partition's windows tile, so the window still sizes every logical axis.
            if comptime!(composition == Composition::Overlapping) {
                out.push(comptime!(space.extent_at(p) as u32));
            } else {
                out.push(bound.at(p));
            }
        }

        out
    }
}

impl TileMatrix {
    /// The last position of `group` whose axis reaches past one in `space`: the axis a row (or a
    /// column) of that group steps.
    fn innermost_past_one(space: &Space, group: core::ops::Range<usize>) -> Option<usize> {
        group.rev().find(|&p| space.extent_at(p) > 1)
    }
}

#[cube]
impl Layout for TileMatrix {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (row, col) = pos;
        let mut coords = self.batches.clone();
        coords.extend(&self.row_extents.unravel(row));
        coords.extend(&self.col_extents.unravel(col));
        coords.to_dyn()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (row, col) = pos;
        let (rows, cols) = self.tile_shape;
        row < rows && col < cols
    }
}

#[cube]
impl Tile {
    /// The `i`-th batch matrix over `axes`, unpacked if packed.
    pub(crate) fn matrix_packed<T: Numeric, W: Size>(
        &self,
        #[comptime] axes: MatrixAxes,
        i: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let g = self.mem("matrix");
        let layout = g.batch_matrix(comptime!(self.place.space.clone()), axes, i);
        g.packed::<T, W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
    }

    /// The fragment's grouped matrix, unpacked if packed; `cols` is in scalars.
    pub(crate) fn fragment_matrix_packed<T: Numeric, W: Size>(
        &self,
        #[comptime] rows: usize,
        #[comptime] cols: usize,
    ) -> MatrixView<'_, Vector<T, W>> {
        let g = self.mem("fragment_matrix");
        let layout = g.whole_matrix(comptime!(self.place.space.clone()), rows, cols);
        g.packed::<T, W, Coords2d, ProjectedMatrix>(layout, comptime!(Guard::Checked))
    }
}
