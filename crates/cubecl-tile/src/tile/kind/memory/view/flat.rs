//! The flat 1-D view over a [`Tile`]: [`FlatLayout`] and its masked [`FlatView`].

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// A masked flat row-major scan over a [`Tile`].
pub(crate) type FlatView<'a, T> = Masked<'a, T, Coords1d>;
/// The mutable twin of [`FlatView`].
pub(crate) type FlatViewMut<'a, T> = MaskedMut<'a, T, Coords1d>;

/// Maps a flat row-major index to an N-D coordinate over `shape` ([`unravel`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct FlatLayout {
    shape: Coords<u32>,
}

#[cube]
impl FlatLayout {
    pub(crate) fn new(shape: Coords<u32>) -> Self {
        FlatLayout { shape }
    }
}

#[cube]
impl Layout for FlatLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.shape.unravel(pos as u32).to_dyn()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        let rank = self.shape.len();
        self.shape
            .product(comptime!((0..rank).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.shape()
    }
}

#[cube]
impl Tile {
    /// A flat row-major view over `Vector<T, W>` lines of the tile's window, masking the overhang.
    /// A packed store is refused.
    pub fn flat<T: Numeric, W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        let g = self.mem("flat");
        if comptime!(g.store.packing != Packing::Plain) {
            panic!("Tile::flat: a packed tile only unpacks under Tile::copy_from")
        }
        g.flat::<T, W>()
    }

    /// The mutable twin of [`flat`](Tile::flat).
    pub fn flat_mut<T: Numeric, W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        self.mem_mut("flat_mut").flat_mut::<T, W>()
    }
}
