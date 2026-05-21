use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

/// Maps semantic coordinates (rank R) to physical coordinates (rank R + n).
///
/// For each axis in `[start_axis, start_axis + n)`, the semantic coordinate is
/// split into a grid component and a tile component: `c -> (c / T, c % T)`.
/// The physical layout is `[Pre-axes, Grid-axes, Tile-axes, Post-axes]`.
#[derive(CubeType, Clone)]
pub struct TiledLayout {
    physical_shape: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    tiles: CoordsDyn,
}

#[cube]
impl TiledLayout {
    pub fn new(
        physical_shape: CoordsDyn,
        #[comptime] start_axis: usize,
        tiles: CoordsDyn,
    ) -> TiledLayout {
        TiledLayout {
            physical_shape,
            start_axis,
            tiles,
        }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        #[comptime]
        let n = self.tiles.len();
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut physical = CoordsDyn::new();

        #[unroll]
        for i in 0..self.start_axis {
            physical.push(pos[i]);
        }

        #[unroll]
        for i in 0..n {
            let tile_size = self.tiles[i];
            physical.push(pos[comptime!(self.start_axis + i)] / tile_size);
        }

        #[unroll]
        for i in 0..n {
            let tile_size = self.tiles[i];
            physical.push(pos[comptime!(self.start_axis + i)] % tile_size);
        }

        let post_start = comptime!(self.start_axis + n);
        let physical_post_start = comptime!(self.start_axis + 2 * n);
        #[unroll]
        for i in 0..comptime!(physical_rank - physical_post_start) {
            physical.push(pos[comptime!(post_start + i)]);
        }

        physical
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        #[comptime]
        let n = self.tiles.len();
        #[comptime]
        let physical_rank = self.physical_shape.len();

        let mut semantic = CoordsDyn::new();

        #[unroll]
        for i in 0..self.start_axis {
            semantic.push(self.physical_shape[i]);
        }

        #[unroll]
        for i in 0..n {
            let grid = self.physical_shape[comptime!(self.start_axis + i)];
            let tile = self.physical_shape[comptime!(self.start_axis + n + i)];
            semantic.push(grid * tile);
        }

        let post_start = comptime!(self.start_axis + 2 * n);
        #[unroll]
        for i in 0..comptime!(physical_rank - post_start) {
            semantic.push(self.physical_shape[comptime!(post_start + i)]);
        }

        semantic
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut is_valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            is_valid = is_valid && pos[i] < bounds[i];
        }
        is_valid
    }
}
