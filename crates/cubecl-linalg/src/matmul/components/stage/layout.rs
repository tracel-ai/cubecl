use std::marker::PhantomData;

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

// TODO tmp name
#[cube]
pub trait TilingOrderTrait: 'static + Send + Sync + Clone + Copy {
    // TODO tmp name, see older commits
    fn a(nth: u32, #[comptime] num_x: u32, #[comptime] num_y: u32) -> (u32, u32);
    // TODO tmp name, see older commits
    fn b(x: u32, y: u32, #[comptime] tile_count_x: u32, #[comptime] tile_count_y: u32) -> u32;
}

#[derive(CubeType, Clone, Copy)]
pub struct RowMajorTilingOrder {}
#[derive(CubeType, Clone, Copy)]
pub struct ColMajorTilingOrder {}

#[cube]
impl TilingOrderTrait for RowMajorTilingOrder {
    fn a(nth: u32, #[comptime] _num_x: u32, #[comptime] num_y: u32) -> (u32, u32) {
        (nth / num_y, nth % num_y)
    }
    fn b(x: u32, y: u32, #[comptime] _tile_count_x: u32, #[comptime] tile_count_y: u32) -> u32 {
        x * tile_count_y + y
    }
}

#[cube]
impl TilingOrderTrait for ColMajorTilingOrder {
    fn a(nth: u32, #[comptime] num_x: u32, #[comptime] _num_y: u32) -> (u32, u32) {
        (nth % num_x, nth / num_x)
    }
    fn b(x: u32, y: u32, #[comptime] tile_count_x: u32, #[comptime] _tile_count_y: u32) -> u32 {
        y * tile_count_x + x
    }
}

#[derive(Clone, Copy)]
pub struct ContiguousTilingLayout<T: TilingOrderTrait> {
    tiling_order: PhantomData<T>,
}

#[derive(Clone, Copy)]
pub struct StridedTilingLayout {}

#[cube]
pub trait TilingLayoutTrait: 'static + Send + Sync + Clone + Copy {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage_slice: Slice<Line<ES>>,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES>;
}

#[cube]
impl<T: TilingOrderTrait> ContiguousTilingLayout<T> {
    /// Converts a tile index in the stage to its (x,y) position
    pub fn to_x_y<S: StageConfig>(
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> (u32, u32) {
        let stage_tiling = config.tiling_dimensions(ident);
        let num_x = stage_tiling.tile_count_row();
        let num_y = stage_tiling.tile_count_col();

        T::a(nth, num_x, num_y)
    }
}

#[cube]
impl<T: TilingOrderTrait> TilingLayoutTrait for ContiguousTilingLayout<T> {
    fn get_tile<ES: Numeric, S: StageConfig>(
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

        let start = tile_shape_x * tile_shape_y * T::b(x, y, tile_count_x, tile_count_y);

        Tile::new_contiguous::<S::TmmConfig>(
            stage_slice.slice(start, start + length),
            ident,
            config.to_tmm_config(),
        )
    }
}

#[cube]
impl StridedTilingLayout {
    /// Returns the nth slice of the stage
    pub fn nth_slice<ES: Numeric, S: StageConfig>(
        stage_slice: &mut SliceMut<Line<ES>>,
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> SliceMut<Line<ES>> {
        let tiling_dimensions = config.tiling_dimensions(ident);
        let matrix_layout = config.matrix_layout(ident);
        let line_size = config.line_size(ident);

        let slice_length = match comptime!(matrix_layout) {
            MatrixLayout::RowMajor => tiling_dimensions.total_col(),
            MatrixLayout::ColMajor => tiling_dimensions.total_row(),
        } / line_size;

        let start = slice_length * nth;
        stage_slice.slice_mut(start, start + slice_length)
    }
}

#[cube]
impl TilingLayoutTrait for StridedTilingLayout {
    fn get_tile<ES: Numeric, S: StageConfig>(
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
}
