use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, MatrixLayout};

use super::{Stage, StageConfig};

#[cube]
pub trait TilingOrder: 'static + Send + Sync + Clone + Copy {
    fn to_row_col(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
    ) -> (u32, u32);

    fn to_nth_tile(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
    ) -> u32;
}

#[derive(CubeType, Clone, Copy)]
pub struct RowMajorTilingOrder {}
#[derive(CubeType, Clone, Copy)]
pub struct ColMajorTilingOrder {}

#[cube]
impl TilingOrder for RowMajorTilingOrder {
    fn to_row_col(
        nth: u32,
        #[comptime] _tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
    ) -> (u32, u32) {
        (nth / tile_count_cols, nth % tile_count_cols)
    }
    fn to_nth_tile(
        row: u32,
        col: u32,
        #[comptime] _tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
    ) -> u32 {
        row * tile_count_cols + col
    }
}

#[cube]
impl TilingOrder for ColMajorTilingOrder {
    fn to_row_col(nth: u32, #[comptime] num_rows: u32, #[comptime] _num_cols: u32) -> (u32, u32) {
        (nth % num_rows, nth / num_rows)
    }
    fn to_nth_tile(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] _tile_count_cols: u32,
    ) -> u32 {
        col * tile_count_rows + row
    }
}

#[derive(Clone, Copy)]
pub struct ContiguousTilingLayout<T: TilingOrder> {
    tiling_order: PhantomData<T>,
}

#[derive(Clone, Copy)]
pub struct StridedTilingLayout {}

#[cube]
pub trait TilingLayout: 'static + Send + Sync + Clone + Copy {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage: &Stage<ES, Self>,
        x: u32,
        y: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES>;
}

#[cube]
impl<T: TilingOrder> ContiguousTilingLayout<T> {
    /// Converts a tile index in the stage to its (x,y) position
    pub fn to_x_y<S: StageConfig>(
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> (u32, u32) {
        let stage_tiling = config.tiling_dimensions(ident);
        let num_x = stage_tiling.tile_count_row();
        let num_y = stage_tiling.tile_count_col();

        T::to_row_col(nth, num_x, num_y)
    }
}

#[cube]
impl<T: TilingOrder> TilingLayout for ContiguousTilingLayout<T> {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage: &Stage<ES, Self>,
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

        let start = tile_shape_x * tile_shape_y * T::to_nth_tile(x, y, tile_count_x, tile_count_y);

        Tile::new_contiguous::<S::TmmConfig>(
            stage.as_slice().slice(start, start + length),
            ident,
            config.to_tmm_config(),
        )
    }
}

#[cube]
impl StridedTilingLayout {
    /// Returns the nth slice of the stage
    pub fn nth_slice<ES: Numeric, S: StageConfig>(
        stage: &mut Stage<ES, Self>,
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
        stage.as_slice_mut().slice_mut(start, start + slice_length)
    }
}

#[cube]
impl TilingLayout for StridedTilingLayout {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage: &Stage<ES, Self>,
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

                Tile::new_strided(stage.as_slice().slice(start, start + length), stride)
            }
            MatrixLayout::ColMajor => {
                let tile_shape_x = tiling_dimensions.tile_shape_row() / line_size;
                let tile_shape_y = tiling_dimensions.tile_shape_col();

                let stride = tile_count_x * tile_shape_x;
                let length = (tile_shape_y - 1) * stride + tile_shape_x;
                let start = x * tile_shape_x + y * tile_shape_y * stride;

                Tile::new_strided(stage.as_slice().slice(start, start + length), stride)
            }
        }
    }
}
