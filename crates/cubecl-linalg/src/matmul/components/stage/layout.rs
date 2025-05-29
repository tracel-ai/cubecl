use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::matmul::components::tile::Tile;
use crate::matmul::components::{Ident, InputIdent, MatrixLayout};

use super::{StageConfig, StageMemory};

#[cube]
pub trait TilingOrder: 'static + Send + Sync + Clone + Copy {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
        #[comptime] config: C,
    ) -> (u32, u32);

    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
        #[comptime] config: C,
    ) -> u32;
}

#[derive(CubeType, Clone, Copy)]
/// Tiles laid out in row-major order.
///
/// Each tile is contiguous, and tiles are placed side by side,
/// row by row (left to right, top to bottom).
/// Example tile indices:
///
/// ```text
/// ┌───┬───┐
/// │ 0 │ 1 │
/// ├───┼───┤
/// │ 2 │ 3 │
/// ├───┼───┤
/// │ 4 │ 5 │
/// ├───┼───┤
/// │ 6 │ 7 │
/// └───┴───┘
/// ```
pub struct RowMajorTilingOrder {}

#[derive(CubeType, Clone, Copy)]
/// Tiles laid out in column-major order.
///
/// Each tile is contiguous, and tiles are placed top to bottom,
/// column by column (like reading columns left to right).
///
/// Example tile indices:
///
/// ```text
/// ┌───┬───┐
/// │ 0 │ 4 │
/// ├───┼───┤
/// │ 1 │ 5 │
/// ├───┼───┤
/// │ 2 │ 6 │
/// ├───┼───┤
/// │ 3 │ 7 │
/// └───┴───┘
/// ```
pub struct ColMajorTilingOrder {}

#[derive(CubeType, Clone, Copy)]
/// Tiles are laid out in column-major order across a fixed number of rows,
/// with all tiles from those rows placed contiguously side by side.
///
/// The grouping should match the set of tiles processed by a warp,
/// so warp-local tile memory remains contiguous.
///
/// This layout ensures that for Lhs data, all tiles needed for a given
/// `k` iteration are stored contiguously, before moving to the next iteration.
///
/// Note: use only for Lhs
///
/// Example tile indices for 4 rows grouped 2 at a time:
///
/// ```text
/// ┌───┬───┐
/// │ 0 │ 2 │
/// ├───┼───┤
/// │ 1 │ 3 │
/// ├───┼───┤
/// │ 4 │ 6 │
/// ├───┼───┤
/// │ 5 │ 7 │
/// └───┴───┘
/// ```
pub struct OrderedTilingOrder {}

#[cube]
impl TilingOrder for RowMajorTilingOrder {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] _tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: C,
    ) -> (u32, u32) {
        (nth / tile_count_cols, nth % tile_count_cols)
    }
    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] _tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: C,
    ) -> u32 {
        row * tile_count_cols + col
    }
}

#[cube]
impl TilingOrder for ColMajorTilingOrder {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] num_rows: u32,
        #[comptime] _num_cols: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: C,
    ) -> (u32, u32) {
        (nth % num_rows, nth / num_rows)
    }
    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] _tile_count_cols: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: C,
    ) -> u32 {
        col * tile_count_rows + row
    }
}

#[cube]
impl TilingOrder for OrderedTilingOrder {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
        #[comptime] config: C,
    ) -> (u32, u32) {
        if Ident::Lhs != ident {
            panic!("Ordered tiling order should be used only on Lhs")
        }

        let group_rows = tile_count_rows / config.num_planes();
        let tiles_per_group = group_rows * tile_count_cols;

        let group = nth / tiles_per_group;
        let pos_within_group = nth % tiles_per_group;

        let local_row = pos_within_group % group_rows;
        let row = group * group_rows + local_row;
        let col = pos_within_group / group_rows;

        (row, col)
    }

    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
        #[comptime] config: C,
    ) -> u32 {
        if Ident::Lhs != ident {
            panic!("Ordered tiling order should be used only on Lhs")
        }

        let group_rows = tile_count_rows / config.num_planes();
        let group = row / group_rows;

        let local_row = row % group_rows;
        let tiles_per_group = group_rows * tile_count_cols;
        let pos_within_group = col * group_rows + local_row;

        group * tiles_per_group + pos_within_group
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
        stage: &StageMemory<ES, Self>,
        x: u32,
        y: u32,
        #[comptime] buffer_index: u32,
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

        T::to_row_col::<S>(nth, num_x, num_y, ident, config)
    }
}

#[cube]
impl<TO: TilingOrder> TilingLayout for ContiguousTilingLayout<TO> {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage_memory: &StageMemory<ES, Self>,
        row: u32,
        col: u32,
        #[comptime] buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        let stage_line_size = config.stage_line_size(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let tiling_scheme = config.tiling_scheme();
        let matrix_layout = config.matrix_layout(ident);

        let (row_buffer_offset, col_buffer_offset, total_tile_count_row, total_tile_count_col) =
            match ident.as_input_ident() {
                InputIdent::Lhs => {
                    let x_tile_offset = 0;
                    let y_tile_offset = tiling_dimensions.tile_count_col() * buffer_index;
                    let total_tile_count_x = tiling_dimensions.tile_count_row();
                    let total_tile_count_y =
                        tiling_dimensions.tile_count_col() * config.num_stages(InputIdent::Lhs);
                    (
                        x_tile_offset,
                        y_tile_offset,
                        total_tile_count_x,
                        total_tile_count_y,
                    )
                }
                InputIdent::Rhs => {
                    let x_tile_offset = tiling_dimensions.tile_count_row() * buffer_index;
                    let y_tile_offset = 0;
                    let total_tile_count_x =
                        tiling_dimensions.tile_count_row() * config.num_stages(InputIdent::Rhs);
                    let total_tile_count_y = tiling_dimensions.tile_count_col();
                    (
                        x_tile_offset,
                        y_tile_offset,
                        total_tile_count_x,
                        total_tile_count_y,
                    )
                }
            };

        let (tile_size_x, tile_size_y, tile_slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_size_x = tiling_scheme.elements_in_tile_row(ident);
                let tile_size_y = tiling_scheme.elements_in_tile_col(ident) / stage_line_size;
                let stride_x = comptime!(tile_size_y * total_tile_count_col);
                let length = (tile_size_x - 1) * stride_x + tile_size_y;

                (tile_size_x, tile_size_y, length)
            }
            MatrixLayout::ColMajor => {
                let tile_size_x = tiling_scheme.elements_in_tile_row(ident) / stage_line_size;
                let tile_size_y = tiling_scheme.elements_in_tile_col(ident);
                let stride_y = comptime!(tile_size_x * total_tile_count_row);
                let length = (tile_size_y - 1) * stride_y + tile_size_x;

                (tile_size_x, tile_size_y, length)
            }
        };

        let start = tile_size_x
            * tile_size_y
            * TO::to_nth_tile::<S>(
                row + row_buffer_offset,
                col + col_buffer_offset,
                total_tile_count_row,
                total_tile_count_col,
                ident,
                config,
            );

        Tile::new_contiguous::<S::TmmConfig>(
            stage_memory
                .as_slice(stage_line_size)
                .slice(start, start + tile_slice_length),
            ident,
            config.to_tmm_config(),
        )
    }
}

#[cube]
impl StridedTilingLayout {
    /// Returns the nth slice of the stage
    pub fn nth_slice<ES: Numeric, S: StageConfig>(
        stage: &mut StageMemory<ES, Self>,
        nth: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> SliceMut<Line<ES>> {
        let matrix_layout = config.matrix_layout(ident);
        let stage_line_size = config.stage_line_size(ident);

        let slice_length = match comptime!(matrix_layout) {
            MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_col(ident),
            MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_row(ident),
        } / stage_line_size;

        let start = slice_length * nth;
        stage
            .as_slice_mut(stage_line_size)
            .slice_mut(start, start + slice_length)
    }
}

#[cube]
impl TilingLayout for StridedTilingLayout {
    fn get_tile<ES: Numeric, S: StageConfig>(
        stage: &StageMemory<ES, Self>,
        x: u32,
        y: u32,
        #[comptime] _buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: S,
    ) -> Tile<ES> {
        if comptime!(config.num_stages(ident.as_input_ident()) > 1) {
            unimplemented!()
        }

        let stage_line_size = config.stage_line_size(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let tiling_scheme = config.tiling_scheme();
        let matrix_layout = config.matrix_layout(ident);

        let tile_count_x = tiling_dimensions.tile_count_row();
        let tile_count_y = tiling_dimensions.tile_count_col();

        match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_size_x = tiling_scheme.elements_in_tile_row(ident);
                let tile_size_y = tiling_scheme.elements_in_tile_col(ident) / stage_line_size;

                let stride = tile_count_y * tile_size_y;
                let length = (tile_size_x - 1) * stride + tile_size_y;
                let start = x * tile_size_x * stride + y * tile_size_y;

                Tile::new_strided(
                    stage.as_slice(stage_line_size).slice(start, start + length),
                    stride,
                    matrix_layout,
                )
            }
            MatrixLayout::ColMajor => {
                let tile_size_x = tiling_scheme.elements_in_tile_row(ident) / stage_line_size;
                let tile_size_y = tiling_scheme.elements_in_tile_col(ident);

                let stride = tile_count_x * tile_size_x;
                let length = (tile_size_y - 1) * stride + tile_size_x;
                let start = x * tile_size_x + y * tile_size_y * stride;

                Tile::new_strided(
                    stage.as_slice(stage_line_size).slice(start, start + length),
                    stride,
                    matrix_layout,
                )
            }
        }
    }
}
