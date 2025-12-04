use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::{Coords2d, Layout, LayoutExpand};

use crate::components::{MatrixLayout, StageIdent, stage::StageMemoryConfig};

pub type TiledCoords = (Coords2d, u32);

/// Tiling mapping on a 2D layout. Unit offset is translated to a 2D offset within the tile.
#[derive(CubeType)]
pub struct TiledLayout {
    #[cube(comptime)]
    ident: StageIdent,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl TiledLayout {
    pub fn new(#[comptime] ident: StageIdent, #[comptime] config: StageMemoryConfig) -> Self {
        TiledLayout { ident, config }
    }
}

#[cube]
impl Layout for TiledLayout {
    type Coordinates = TiledCoords;
    type SourceCoordinates = Coords2d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (tile, unit_pos) = pos;
        let (tile_row, tile_col) = tile;

        let tile_size_row = comptime![self.config.elements_per_tile_along_row];
        let tile_size_col = comptime![self.config.elements_per_tile_along_col];

        let view_tile_row = tile_row * tile_size_row;
        let view_tile_col = tile_col * tile_size_col;

        let (unit_row, unit_col) = match comptime![self.config.matrix_layout] {
            MatrixLayout::RowMajor => (unit_pos / tile_size_col, unit_pos % tile_size_col),
            MatrixLayout::ColMajor => (unit_pos % tile_size_row, unit_pos / tile_size_row),
        };

        (view_tile_row + unit_row, view_tile_col + unit_col)
    }

    fn shape(&self) -> Self::Coordinates {
        let config = comptime![self.config];
        let tile_size_row = config.elements_per_tile_along_row;
        let tile_size_col = config.elements_per_tile_along_col;

        let tiles_row = config.elements_per_stage_along_row() / tile_size_row;
        let tiles_col = config.elements_per_stage_along_col() / tile_size_col;
        let tile_size = tile_size_row * tile_size_col;

        let (tiles_row, tiles_col) = match comptime![self.ident] {
            StageIdent::Lhs => (tiles_row, tiles_col * config.num_stages).runtime(),
            StageIdent::Rhs => (tiles_row * config.num_stages, tiles_col).runtime(),
            StageIdent::Acc => (tiles_row, tiles_col).runtime(),
            StageIdent::Out => (tiles_row, tiles_col).runtime(),
        };

        ((tiles_row, tiles_col), tile_size)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        // Bounds checking should be handled by underlying layout
        true.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}
