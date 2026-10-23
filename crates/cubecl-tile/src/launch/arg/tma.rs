//! A TMA tensor map as an operand's values ([`TileArgLaunch::tensor_map`]) and its in-kernel
//! layouts.

use cubecl::prelude::*;
use cubecl_std::tensor::{
    layout::{CoordsDyn, Layout, LayoutExpand, as_dyn::IntoDynLayout},
    view::launch::VirtualViewLayoutLaunch,
};

use crate::*;

/// The box a TMA descriptor moves: logical `(rows, cols)`, optional batch, and whether the
/// descriptor is transposed (column-major).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmaBox {
    pub rows: u32,
    pub cols: u32,
    pub batch: Option<u32>,
    pub transposed: bool,
}

/// What the TMA family launches from: the tensor map, the axes it spans, and its box.
pub struct TmaOperand {
    pub map: TensorMapArg<Tiled>,
    pub axes: Vec<Axis>,
    pub shape: TmaBox,
}

impl TmaOperand {
    /// The tensor map as an operand's values, served at the element it moves, under no scales.
    pub fn arg(self) -> TileArgLaunch {
        TileArgLaunch::tensor_map(self.map, &self.axes, self.shape)
    }
}

impl TileArgLaunch {
    /// A TMA tensor map as a tile argument over `axes`: two, or three with `shape.batch` set.
    pub fn tensor_map(tensor_map: TensorMapArg<Tiled>, axes: &[Axis], shape: TmaBox) -> Self {
        let layout = Self::dyn_layout(axes, shape);
        Self::of_tensor_map(
            tensor_map,
            VirtualViewLayoutLaunch::new::<IntoDynLayout<TmaDynLayout>>(layout),
            TileSpec::direct(axes),
        )
    }

    /// The layout of a descriptor over `axes`.
    fn dyn_layout(axes: &[Axis], shape: TmaBox) -> TmaDynLayoutLaunch {
        let batched = match (axes.len(), shape.batch) {
            (2, None) => false,
            (3, Some(_)) => true,
            (r, batch) => panic!(
                "TileArgLaunch::tensor_map: the descriptor is (batch, row, col); {r} axes with \
                 batch {batch:?} is not a matrix nor a batched one"
            ),
        };
        let dims = (shape.batch.unwrap_or(1), shape.rows, shape.cols);
        TmaDynLayoutLaunch::new(dims, batched, shape.transposed)
    }
}

/// Where a box of a storage-tiled operand lands its values: the order the descriptor states the
/// operand in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TmaLanding {
    /// One storage tile a box, landing as the contiguous run it is stored as: the descriptor is
    /// the operand as stored, `[.., R/tr, C/tc, tr, tc]`.
    Tiles,
    /// Whole tiles along the rows, landing the box's rows one after another, each a run of its
    /// tiles' rows, as a row-major operand's box would land them: the descriptor is the operand as
    /// `[R/tr, tr, C/tc, tc]`, the tiles' rows nested inside the tiles' columns. Along the columns
    /// a box is whole tiles, or a part of one tile's run that divides it, starting where the box's
    /// own columns do.
    Rows,
}

/// In-kernel tensor-map layout aligning logical [`CoordsDyn`] to the descriptor's
/// `(batch, row, col)`.
#[derive(CubeType, CubeLaunch, Clone)]
pub(crate) struct TmaDynLayout {
    /// Logical `(batch, rows, cols)` of the operand.
    dims: (u32, u32, u32),
    #[cube(comptime)]
    batched: bool,
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl Layout for TmaDynLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (batch, _rows, _cols) = self.dims;
        let mut src = CoordsDyn::new();
        if comptime!(self.batched) {
            // A unit-batch descriptor is a broadcast: always read batch 0.
            src.push(select(batch == 1, 0u32, pos[0]));
        } else {
            src.push(0u32);
        }
        let (r, c) = comptime!(if self.batched { (1, 2) } else { (0, 1) });
        // TMA discards the last stride, so a col-major descriptor is transposed; swap back.
        if comptime!(self.transposed) {
            src.push(pos[c]);
            src.push(pos[r]);
        } else {
            src.push(pos[r]);
            src.push(pos[c]);
        }
        src
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        (self.to_source_pos(pos), true)
    }

    fn shape(&self) -> Self::Coordinates {
        let (batch, rows, cols) = self.dims;
        let mut s = CoordsDyn::new();
        if comptime!(self.batched) {
            s.push(batch);
        }
        s.push(rows);
        s.push(cols);
        s
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
    }
}
