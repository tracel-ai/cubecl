use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::{MatrixLayout, global::memory::GlobalMemoryConfig};

/// Reads data from the tensor view as a window, i.e. a slice of global memory
/// Also returns the length of the slice
///
/// The length of the slice is the width of the tile
///
/// # Note
///
/// If the slice would be partly out-of-bounds, it will simply be shorter.
/// The caller must do the padding if necessary.
#[cube]
pub fn load_window_in_tile<EG: Numeric>(
    view: &View<Line<EG>, Coords2d>,
    tile: Coords2d,
    nth_window: u32,
    #[comptime] config: GlobalMemoryConfig,
) -> Slice<Line<EG>> {
    let (tile_row, tile_col) = tile;
    let tile_size_row = config.elements_in_tile_row();
    let tile_size_col = config.elements_in_tile_col();

    let size = match config.matrix_layout() {
        MatrixLayout::RowMajor => (1u32, tile_size_col).runtime(),
        MatrixLayout::ColMajor => (tile_size_row, 1u32).runtime(),
    };

    let offset = (tile_row * tile_size_row, tile_col * tile_size_col);
    let tile_size = (tile_size_row, tile_size_col).runtime();

    load_window(&view.slice(offset, tile_size), nth_window, size, config)
}

/// Reads data from the tensor view as a window, i.e. a slice of global memory
///
/// The length of the slice is the width of the tile
///
/// # Note
///
/// If the slice would be partly out-of-bounds, it will simply be shorter.
/// The caller must do the padding if necessary.
#[cube]
pub fn load_window_in_stage<EG: Numeric>(
    view: &View<Line<EG>, Coords2d>,
    nth_window: u32,
    #[comptime] config: GlobalMemoryConfig,
) -> Slice<Line<EG>> {
    let size = match config.matrix_layout() {
        MatrixLayout::RowMajor => (1u32, config.elements_in_stage_col()).runtime(),
        MatrixLayout::ColMajor => (config.elements_in_stage_row(), 1u32).runtime(),
    };

    load_window(view, nth_window, size, config)
}

#[cube]
fn load_window<EG: Numeric>(
    view: &View<Line<EG>, Coords2d>,
    nth_window: u32,
    size: Coords2d,
    #[comptime] config: GlobalMemoryConfig,
) -> Slice<Line<EG>> {
    let offset = match config.matrix_layout() {
        MatrixLayout::RowMajor => (nth_window, 0),
        MatrixLayout::ColMajor => (0, nth_window),
    };

    if comptime![config.check_row_bounds() || config.check_col_bounds()] {
        view.slice(offset, size).to_linear_slice()
    } else {
        view.slice_unchecked(offset, size).to_linear_slice()
    }
}
