use std::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::layout::Coords2d;

use crate::components::tile::StridedTile;
use crate::components::{InvalidConfigError, stage::StageMemoryConfig};
use crate::components::{MatrixLayout, stage::SwizzleMode};

use super::StridedStageMemory;

#[cube]
/// Determines the order in which tiles are stored in shared memory,
/// if [TilingLayout] is contiguous
pub trait TilingOrder: 'static + Send + Sync + Clone + Copy {
    /// Returns the coordinates (row, col) of the tile
    fn to_row_col(
        nth: u32,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Coords2d;

    /// Given the coordinates (row, col) of the tile,
    /// returns its index in shared memory
    fn to_nth_tile(
        tile: Coords2d,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> u32;

    /// Return the trait value as enum
    fn to_enum() -> comptime_type!(TilingOrderEnum);
}

/// Enum for the available traits
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TilingOrderEnum {
    /// Tiles of the same row are side by side
    RowMajor,
    /// Tiles of the column are side by side
    ColMajor,
    /// Tiles are laid out in column-major order across a fixed number of rows,
    /// with all tiles from those rows placed contiguously side by side.
    Ordered,
    /// If the matrix data layout is row-major, the tiling order is col-major
    /// If the matrix data layout is col-major, the tiling order is row-major
    Tma,
}

/// Enum for the available traits
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TilingLayoutEnum {
    Strided,
    Contiguous(TilingOrderEnum),
    Other,
}

/// Overall config for tiling layout
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TilingLayoutConfig {
    pub lhs: TilingLayoutEnum,
    pub rhs: TilingLayoutEnum,
    pub acc: TilingLayoutEnum,
    pub out: TilingLayoutEnum,
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
    fn to_row_col(
        nth: u32,
        _tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] _config: StageMemoryConfig,
    ) -> Coords2d {
        (nth / tile_count_cols, nth % tile_count_cols)
    }
    fn to_nth_tile(
        tile: Coords2d,
        _tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] _config: StageMemoryConfig,
    ) -> u32 {
        let (row, col) = tile;
        row * tile_count_cols + col
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::RowMajor
    }
}

#[cube]
impl TilingOrder for ColMajorTilingOrder {
    fn to_row_col(
        nth: u32,
        num_rows: u32,
        _num_cols: u32,
        #[comptime] _config: StageMemoryConfig,
    ) -> Coords2d {
        (nth % num_rows, nth / num_rows)
    }
    fn to_nth_tile(
        tile: Coords2d,
        tile_count_rows: u32,
        _tile_count_cols: u32,
        #[comptime] _config: StageMemoryConfig,
    ) -> u32 {
        let (row, col) = tile;
        col * tile_count_rows + row
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::ColMajor
    }
}

#[cube]
impl TilingOrder for OrderedTilingOrder {
    fn to_row_col(
        nth: u32,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Coords2d {
        let group_rows = tile_count_rows / config.num_planes;
        let tiles_per_group = group_rows * tile_count_cols;

        let group = nth / tiles_per_group;
        let pos_within_group = nth % tiles_per_group;

        let local_row = pos_within_group % group_rows;
        let row = group * group_rows + local_row;
        let col = pos_within_group / group_rows;

        (row, col)
    }

    fn to_nth_tile(
        tile: Coords2d,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> u32 {
        let (row, col) = tile;

        let group_rows = tile_count_rows / config.num_planes;
        let group = row / group_rows;

        let local_row = row % group_rows;
        let tiles_per_group = group_rows * tile_count_cols;
        let pos_within_group = col * group_rows + local_row;

        group * tiles_per_group + pos_within_group
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::Ordered
    }
}

#[derive(CubeType, Clone, Copy)]
/// A special tiling order where:
/// - If the matrix data layout is row-major, the tiling order is col-major
/// - If the matrix data layout is col-major, the tiling order is row-major
pub struct TmaTilingOrder;

#[cube]
impl TilingOrder for TmaTilingOrder {
    fn to_row_col(
        nth: u32,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Coords2d {
        match config.matrix_layout {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_row_col(nth, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_row_col(nth, tile_count_rows, tile_count_cols, config)
            }
        }
    }

    fn to_nth_tile(
        tile: Coords2d,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> u32 {
        match config.matrix_layout {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_nth_tile(tile, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_nth_tile(tile, tile_count_rows, tile_count_cols, config)
            }
        }
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::Tma
    }
}

#[cube]
/// Describes how tiles are arranged in shared memory.
pub trait TilingLayout: 'static + Send + Sync + Clone + Copy + TilingValidation {
    /// Returns the tile at shared memory coordinates
    fn get_tile<ES: Numeric>(
        stage: &StridedStageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES>;

    /// Return the trait value as enum
    fn to_enum() -> comptime_type!(TilingLayoutEnum);
}

pub trait TilingValidation {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError>;
}

#[derive(Clone, Copy)]
/// Each tile is stored contiguously in shared memory.
/// Global memory loads may require remapping to match this layout.
pub struct ContiguousTilingLayout<T: TilingOrder> {
    tiling_order: PhantomData<T>,
}

/// TMA tiling depends on the swizzle settings
#[derive(Clone, Copy)]
pub struct TmaTilingLayout {}

#[derive(Clone, Copy)]
/// Tiles follow a strided layout that often mirrors global memory layout.
/// Not all tiles are contiguous in shared memory, but mapping is more direct.
pub struct StridedTilingLayout {}

#[cube]
impl<T: TilingOrder> ContiguousTilingLayout<T> {
    /// Converts a tile index in the stage to its (x,y) position
    pub fn to_x_y(nth: u32, #[comptime] config: StageMemoryConfig) -> Coords2d {
        let num_x = config.tiles_per_stage_along_row();
        let num_y = config.tiles_per_stage_along_col();

        T::to_row_col(nth, num_x, num_y, config)
    }
}

#[cube]
impl<TO: TilingOrder> TilingLayout for ContiguousTilingLayout<TO> {
    fn get_tile<ES: Numeric>(
        stage_memory: &StridedStageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        let (row, col) = tile;

        let stage_line_size = config.line_size;
        let matrix_layout = config.matrix_layout;

        let (tile_size_x, tile_size_y) = match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_size_x = config.elements_per_tile_along_row;
                let tile_size_y = config.elements_per_tile_along_col / stage_line_size;

                (tile_size_x, tile_size_y)
            }
            MatrixLayout::ColMajor => {
                let tile_size_x = config.elements_per_tile_along_row / stage_line_size;
                let tile_size_y = config.elements_per_tile_along_col;

                (tile_size_x, tile_size_y)
            }
        };

        let start = tile_size_x
            * tile_size_y
            * TO::to_nth_tile(
                (row, col),
                config.tiles_per_stage_along_row(),
                config.tiles_per_stage_along_col(),
                config,
            );

        StridedTile::new_contiguous(stage_memory.as_slice(stage_line_size), start, config)
    }

    fn to_enum() -> comptime_type!(TilingLayoutEnum) {
        comptime![TilingLayoutEnum::Contiguous(TO::to_enum())]
    }
}

impl<TO: TilingOrder> TilingValidation for ContiguousTilingLayout<TO> {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let tile_width = config.elements_per_tile_along_contiguous_dim();
        if config.line_size > tile_width {
            return Err(Box::new("Invalid line size"));
        }
        Ok(())
    }
}

#[cube]
impl StridedTilingLayout {
    /// Returns the nth slice of the stage
    pub fn nth_slice<ES: Numeric>(
        stage: &mut StridedStageMemory<ES, Self>,
        nth: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> SliceMut<Line<ES>> {
        let matrix_layout = config.matrix_layout;
        let stage_line_size = config.line_size;

        let slice_length = match comptime!(matrix_layout) {
            MatrixLayout::RowMajor => config.elements_per_stage_along_col(),
            MatrixLayout::ColMajor => config.elements_per_stage_along_row(),
        } / stage_line_size;

        let start = slice_length * nth;
        stage
            .as_slice_mut(stage_line_size)
            .slice_mut(start, start + slice_length)
    }
}

#[cube]
impl TilingLayout for StridedTilingLayout {
    fn get_tile<ES: Numeric>(
        stage: &StridedStageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        let (x, y) = tile;

        let stage_line_size = config.line_size;
        let matrix_layout = config.matrix_layout;

        let tile_count_x = config.tiles_per_stage_along_row();
        let tile_count_y = config.tiles_per_stage_along_col();

        match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_size_x = config.elements_per_tile_along_row;
                let tile_size_y = config.elements_per_tile_along_col / stage_line_size;

                let stride = tile_count_y * tile_size_y;
                let length = (tile_size_x - 1) * stride + tile_size_y;
                let start = x * tile_size_x * stride + y * tile_size_y;

                StridedTile::new_strided(
                    stage.as_slice(stage_line_size),
                    start,
                    start + length,
                    stride,
                    stage.swizzle,
                    matrix_layout,
                    stage_line_size,
                )
            }
            MatrixLayout::ColMajor => {
                let tile_size_x = config.elements_per_tile_along_row / stage_line_size;
                let tile_size_y = config.elements_per_tile_along_col;

                let stride = tile_count_x * tile_size_x;
                let length = (tile_size_y - 1) * stride + tile_size_x;
                let start = x * tile_size_x + y * tile_size_y * stride;

                StridedTile::new_strided(
                    stage.as_slice(stage_line_size),
                    start,
                    start + length,
                    stride,
                    stage.swizzle,
                    matrix_layout,
                    stage_line_size,
                )
            }
        }
    }

    fn to_enum() -> comptime_type!(TilingLayoutEnum) {
        comptime![TilingLayoutEnum::Strided]
    }
}

impl TilingValidation for StridedTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_per_stage_along_contiguous_dim();
        if config.line_size > stage_width {
            return Err(Box::new("Invalid line size"));
        }
        Ok(())
    }
}

#[cube]
impl TilingLayout for TmaTilingLayout {
    fn get_tile<ES: Numeric>(
        stage: &StridedStageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        match config.swizzle {
            SwizzleMode::None => ContiguousTilingLayout::<TmaTilingOrder>::get_tile(
                &stage.with_layout(),
                tile,
                config,
            ),
            _ => StridedTilingLayout::get_tile(&stage.with_layout(), tile, config),
        }
    }

    fn to_enum() -> comptime_type!(TilingLayoutEnum) {
        comptime![TilingLayoutEnum::Other]
    }
}

impl TilingValidation for TmaTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        match config.swizzle {
            SwizzleMode::None => ContiguousTilingLayout::<TmaTilingOrder>::check(config)?,
            _ => StridedTilingLayout::check(config)?,
        };
        Ok(())
    }
}

#[derive(Clone, Copy)]
/// Dummy tiling layout that panics if it's used. Can be used when the reader is known to be a
/// `FillReader`
pub struct NoTilingLayout {}

#[cube]
impl TilingLayout for NoTilingLayout {
    fn get_tile<ES: Numeric>(
        _stage: &StridedStageMemory<ES, Self>,
        _tile: Coords2d,
        #[comptime] _config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        panic!("Can't get tile of layoutless tiling!")
    }

    fn to_enum() -> comptime_type!(TilingLayoutEnum) {
        comptime![TilingLayoutEnum::Other]
    }
}

impl TilingValidation for NoTilingLayout {
    fn check(_config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}
