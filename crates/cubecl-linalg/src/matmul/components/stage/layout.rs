use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, MatrixLayout};

use super::StageConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// How the tiles are stored in shared memory
pub enum TilingLayout {
    /// Each tile is stored contiguously in memory.
    /// Tiles are placed sequentially in memory according to the specified `TilingOrder`.
    Contiguous(TilingOrder),

    /// Tiles follow the memory layout of the underlying global memory,
    /// meaning elements within a tile may be interleaved with elements from other tiles.
    Strided,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout in which to store tiles within the stage
pub enum TilingOrder {
    /// Tiles are conceptually stored in row-major order, regardless of the actual data layout.
    RowMajor,
    /// Tiles are conceptually stored in column-major order, regardless of the actual data layout.
    ColMajor,
}

#[cube]
impl TilingLayout {
    /// Converts a tile index in the stage to its (x,y) position
    /// Only for contiguous tiling layout
    /// TODO: rethink abstraction
    pub fn to_x_y<S: StageConfig>(
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> (u32, u32) {
        let stage_tiling = config.tiling_dimensions(ident);
        let num_x = stage_tiling.tile_count_row();
        let num_y = stage_tiling.tile_count_col();

        if let TilingLayout::Strided = comptime!(config.tiling_layout(ident)) {
            comptime!(panic!(
                "to_x_y only makes sense in contiguous tiling layout"
            ));
        };

        match comptime!(config.tiling_layout(ident)) {
            TilingLayout::Contiguous(tiling_order) => match comptime!(tiling_order) {
                TilingOrder::RowMajor => (nth / num_y, nth % num_y),
                TilingOrder::ColMajor => (nth % num_x, nth / num_x),
            },
            TilingLayout::Strided => (0u32, 0u32).runtime(), // unreachable
        }
    }

    /// Returns the smallest slice that encompasses all tile elements  
    /// of the tile at (x, y) position.  
    pub fn get_tile<ES: Numeric, S: StageConfig>(
        stage_slice: Slice<Line<ES>>,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        match config.tiling_layout(ident) {
            TilingLayout::Contiguous(tiling_order) => TilingLayout::get_tile_contiguous::<ES, S>(
                stage_slice,
                x,
                y,
                tiling_order,
                ident,
                config,
            ),
            TilingLayout::Strided => {
                TilingLayout::get_tile_strided::<ES, S>(stage_slice, x, y, ident, config)
            }
        }
    }

    pub fn get_tile_contiguous<ES: Numeric, S: StageConfig>(
        stage_slice: Slice<Line<ES>>,
        x: u32,
        y: u32,
        #[comptime] tiling_order: TilingOrder,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let line_size = config.line_size(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let matrix_layout = config.matrix_layout(ident);

        let tile_count_x = tiling_dimensions.tile_count_row();
        let tile_count_y = tiling_dimensions.tile_count_col();

        let (tile_shape_x, tile_shape_y, length) = match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_shape_x = tiling_dimensions.tile_shape_row();
                let tile_shape_y = tiling_dimensions.tile_shape_col() / line_size;
                let stride_x = tile_shape_y;
                let length = (tile_shape_x - 1) * stride_x + tile_shape_y;
                (tile_shape_x, tile_shape_y, length)
            }
            MatrixLayout::ColMajor => {
                let tile_shape_x = tiling_dimensions.tile_shape_row() / line_size;
                let tile_shape_y = tiling_dimensions.tile_shape_col();
                let stride_y = tile_shape_x;
                let length = (tile_shape_y - 1) * stride_y + tile_shape_x;
                (tile_shape_x, tile_shape_y, length)
            }
        };

        let start = tile_shape_x
            * tile_shape_y
            * match comptime!(tiling_order) {
                TilingOrder::RowMajor => x * tile_count_y + y,
                TilingOrder::ColMajor => y * tile_count_x + x,
            };

        Tile::new_contiguous::<S::TmmConfig>(
            stage_slice.slice(start, start + length),
            ident,
            config.to_tmm_config(),
        )
    }

    pub fn get_tile_strided<ES: Numeric, S: StageConfig>(
        stage_slice: Slice<Line<ES>>,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let line_size = config.line_size(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let matrix_layout = config.matrix_layout(ident);

        let tile_count_x = tiling_dimensions.tile_count_row();
        let tile_count_y = tiling_dimensions.tile_count_col();

        match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_shape_x = tiling_dimensions.tile_shape_row();
                let tile_shape_y = tiling_dimensions.tile_shape_col() / line_size;

                let stride = tile_count_y * tile_shape_y;
                let length = (tile_shape_x - 1) * stride + tile_shape_y;
                let start = x * tile_shape_x * stride + y * tile_shape_y;

                Tile::new_strided(stage_slice.slice(start, start + length), stride)
            }
            MatrixLayout::ColMajor => {
                let tile_shape_x = tiling_dimensions.tile_shape_row() / line_size;
                let tile_shape_y = tiling_dimensions.tile_shape_col();

                let stride = tile_count_x * tile_shape_x;
                let length = (tile_shape_y - 1) * stride + tile_shape_x;
                let start = x * tile_shape_x + y * tile_shape_y * stride;

                Tile::new_strided(stage_slice.slice(start, start + length), stride)
            }
        }
    }

    /// Returns the nth slice of the stage
    /// Only for strided layout
    /// TODO: rethink abstractions
    pub fn nth_slice<ES: Numeric, S: StageConfig>(
        stage_slice: &mut SliceMut<Line<ES>>,
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> SliceMut<Line<ES>> {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let matrix_layout = config.matrix_layout(ident);
        let tiling_layout = config.tiling_layout(ident);
        let line_size = config.line_size(ident);

        if let TilingLayout::Contiguous(_) = tiling_layout {
            comptime!(panic!("nth slice only makes sense in strided layout"));
        }

        let slice_length = match comptime!(matrix_layout) {
            MatrixLayout::RowMajor => tiling_dimensions.total_col(),
            MatrixLayout::ColMajor => tiling_dimensions.total_row(),
        } / line_size;

        let start = slice_length * nth;
        stage_slice.slice_mut(start, start + slice_length)
    }
}
