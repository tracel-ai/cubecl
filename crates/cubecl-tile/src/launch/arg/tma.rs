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

impl TileArgLaunch {
    /// A TMA tensor map as a tile argument over `axes`, `extents` along each, whose descriptor
    /// dims are not the tile's own axes: each descriptor coordinate is read off a position over
    /// `axes` as `coordinates` says, as a block-split operand lands in a descriptor over the
    /// matrix it splits (its `KB` and `KI` one column).
    ///
    /// # Panics
    ///
    /// Where `extents` does not give one extent an axis, or a term reads past `axes`.
    pub fn tensor_map_folded(
        tensor_map: TensorMapArg<Tiled>,
        axes: &[Axis],
        extents: &[u32],
        coordinates: TmaCoordinates,
    ) -> Self {
        assert_eq!(
            extents.len(),
            axes.len(),
            "TileArgLaunch::tensor_map_folded: one extent an axis, {axes:?} against {extents:?}"
        );
        assert!(
            coordinates
                .dims
                .iter()
                .flatten()
                .all(|term| term.position < axes.len()),
            "TileArgLaunch::tensor_map_folded: a term reads a position past {axes:?}"
        );
        let mut shape = SequenceArg::new();
        for &extent in extents {
            shape.push(extent);
        }
        let layout = TmaFoldedLayoutLaunch::new(shape, coordinates);
        Self::of_tensor_map(
            tensor_map,
            VirtualViewLayoutLaunch::new::<IntoDynLayout<TmaFoldedLayout>>(layout),
            TileSpec::direct(axes),
        )
    }
}

/// One term of a descriptor coordinate: the position along the tile's `position`-th axis, times
/// `scale`, modulo `modulus` where it is not zero, over `divisor`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmaTerm {
    pub position: usize,
    pub scale: u32,
    pub modulus: u32,
    pub divisor: u32,
}

impl TmaTerm {
    /// The position along the `position`-th axis as it is.
    pub fn axis(position: usize) -> Self {
        Self {
            position,
            scale: 1,
            modulus: 0,
            divisor: 1,
        }
    }

    /// This term times `scale`.
    pub fn times(self, scale: u32) -> Self {
        Self { scale, ..self }
    }

    /// This term over `divisor`: a position the box's origin only ever takes at multiples of it.
    pub fn over(self, divisor: u32) -> Self {
        Self { divisor, ..self }
    }

    /// This term modulo `modulus`, before its divisor: the place of a box inside a stored tile a
    /// box covers part of.
    pub fn modulo(self, modulus: u32) -> Self {
        Self { modulus, ..self }
    }
}

/// How a tile's position lands in a descriptor: one sum of [`TmaTerm`]s a descriptor dim,
/// outermost first, the order the descriptor states its dims in.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TmaCoordinates {
    pub dims: Vec<Vec<TmaTerm>>,
}

/// In-kernel tensor-map layout landing a tile's position, over its own axes, in a descriptor whose
/// dims sum them ([`TmaCoordinates`]).
#[derive(CubeType, CubeLaunch, Clone)]
pub(crate) struct TmaFoldedLayout {
    /// The extent of each of the tile's axes.
    shape: CoordsDyn,
    #[cube(comptime)]
    coordinates: TmaCoordinates,
}

#[cube]
impl Layout for TmaFoldedLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let coordinates = comptime!(self.coordinates.clone());
        let mut src = CoordsDyn::new();
        #[unroll]
        for d in 0..comptime!(coordinates.dims.len()) {
            let terms = comptime!(coordinates.dims[d].clone());
            let mut coordinate = 0u32;
            // Indexed: an unrolled loop walks a comptime list by its positions.
            #[allow(clippy::needless_range_loop)]
            #[unroll]
            for t in 0..comptime!(terms.len()) {
                let term = comptime!(terms[t]);
                let mut value = pos[comptime!(term.position)] * comptime!(term.scale);
                if comptime!(term.modulus != 0) {
                    value %= comptime!(term.modulus);
                }
                coordinate += value / comptime!(term.divisor);
            }
            src.push(coordinate);
        }
        src
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        (self.to_source_pos(pos), true)
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape.clone()
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
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
