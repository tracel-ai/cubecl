use crate::components::MatmulIdent;
use crate::components::MatrixLayout;
use crate::components::global::memory::GlobalMemoryConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords3d};

#[derive(Clone, CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorReader<EI: Numeric> {
    pub view: View<Line<EI>, Coords3d>,
    pub row_offset: RuntimeCell<u32>,
    pub col_offset: RuntimeCell<u32>,
    pub batch_offset: u32,
}

unsafe impl<EG: Numeric> Sync for TensorReader<EG> {}
unsafe impl<EG: Numeric> Send for TensorReader<EG> {}

#[derive(CubeType)]
/// Contiguous slice wrapper for memcpy_async loading
pub struct Window<EG: Numeric> {
    /// Contiguous slice containing all and only data of window
    pub slice: Slice<Line<EG>>,
    /// Number of lines
    pub size: u32,
}

#[cube]
impl<EG: Numeric> TensorReader<EG> {
    /// Instantiate a read view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(view: View<Line<EG>, Coords3d>, offset_global: Coords3d) -> Self {
        let (b, row, col) = offset_global;
        TensorReader::<EG> {
            view,
            row_offset: RuntimeCell::new(row),
            col_offset: RuntimeCell::new(col),
            batch_offset: b,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&self, k_offset: u32, #[comptime] ident: MatmulIdent) {
        match ident {
            MatmulIdent::Lhs => {
                self.col_offset.store(self.col_offset.read() + k_offset);
            }
            MatmulIdent::Rhs => {
                self.row_offset.store(self.row_offset.read() + k_offset);
            }
            MatmulIdent::Out => comptime!(unreachable!()),
        }
    }

    /// Reads data from the tensor view as a window, i.e. a slice of global memory
    /// Also returns the length of the slice
    ///
    /// The length of the slice is the width of the tile
    ///
    /// # Note
    ///
    /// If the slice would be partly out-of-bounds, it will simply be shorter.
    /// The caller must do the padding if necessary.
    pub fn load_window_in_tile(
        &self,
        tile: (u32, u32),
        nth_window: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Window<EG> {
        let line_size = config.global_line_size;
        let matrix_layout = config.matrix_layout;

        let tile_size_x = config.elements_in_tile_row;
        let tile_size_y = config.elements_in_tile_col;

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor => tile_size_y / line_size,
            MatrixLayout::ColMajor => tile_size_x / line_size,
        }};

        self.load_window(
            nth_window,
            (tile.0 * tile_size_x, tile.1 * tile_size_y),
            num_lines_in_window,
            config,
        )
    }

    /// Reads data from the tensor view as a window, i.e. a slice of global memory
    ///
    /// The length of the slice is the width of the tile
    ///
    /// # Note
    ///
    /// If the slice would be partly out-of-bounds, it will simply be shorter.
    /// The caller must do the padding if necessary.
    pub fn load_window_in_stage(
        &self,
        nth_window: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Window<EG> {
        let line_size = config.global_line_size;
        let matrix_layout = config.matrix_layout;

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor =>
                config.elements_in_stage_col / line_size
            ,
            MatrixLayout::ColMajor =>
                config.elements_in_stage_row / line_size
            ,
        }};

        self.load_window(
            nth_window,
            (0u32, 0u32).runtime(),
            num_lines_in_window,
            config,
        )
    }

    fn load_window(
        &self,
        nth_window: u32,
        tile_offsets: (u32, u32),
        #[comptime] num_lines_in_window: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Window<EG> {
        let line_size = config.global_line_size;
        let matrix_layout = config.matrix_layout;

        let (load_row, load_col) = match matrix_layout {
            MatrixLayout::RowMajor => (nth_window, 0),
            MatrixLayout::ColMajor => (0, nth_window),
        };

        let view_tile_row = tile_offsets.0 + self.row_offset.read();
        let view_tile_col = tile_offsets.1 + self.col_offset.read();

        let view_row = view_tile_row + load_row;
        let view_col = view_tile_col + load_col;

        let (_, rows, columns) = self.view.shape();

        let (check_h_bounds, view_h, shape_h, check_w_bounds, view_w, shape_w) =
            match config.matrix_layout {
                MatrixLayout::RowMajor => (
                    config.check_row_bounds,
                    view_row,
                    rows,
                    config.check_col_bounds,
                    view_col,
                    columns,
                ),
                MatrixLayout::ColMajor => (
                    config.check_col_bounds,
                    view_col,
                    columns,
                    config.check_row_bounds,
                    view_row,
                    rows,
                ),
            };

        // There are 0 lines if out-of-bounds vertically
        let max_lines_in_window = if comptime!(check_h_bounds) {
            num_lines_in_window * u32::cast_from(view_h < shape_h)
        } else {
            num_lines_in_window.runtime()
        };

        // Window is clamped if partially out-of-bounds horizontally
        let size = if comptime!(check_w_bounds) {
            slice_length_clamp(shape_w / line_size, view_w / line_size, max_lines_in_window)
        } else {
            max_lines_in_window
        };

        Window::<EG> {
            slice: self
                .view
                .slice((self.batch_offset, view_row, view_col), size),
            size,
        }
    }

    /// Reads data from the tensor view at the specified tile coordinates (tile_x, tile_y).
    ///
    /// Each unit loads one line in a coalesced manner for improved efficiency.
    /// For row-major tensors, subsequent units read lines horizontally within the tile,
    /// while for column-major tensors, they read lines vertically.
    ///
    /// # Note
    ///
    /// Out-of-bounds reads will be translated to zeros.
    pub fn load_coalesced_in_tile(
        &self,
        tile_x: u32,
        tile_y: u32,
        position: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Line<EG> {
        let tile_size_x = config.elements_in_tile_row;
        let tile_size_y = config.elements_in_tile_col;

        let view_tile_x = tile_x * tile_size_x;
        let view_tile_y = tile_y * tile_size_y;

        let (load_x, load_y) = match config.matrix_layout {
            MatrixLayout::RowMajor => (position / tile_size_y, position % tile_size_y),
            MatrixLayout::ColMajor => (position % tile_size_x, position / tile_size_x),
        };

        self.load_coalesced((load_x + view_tile_x, load_y + view_tile_y))
    }

    /// Reads data from the tensor view at the specified index within the whole view,
    /// without regards to tiles
    ///
    /// Each unit loads one line in a coalesced manner for improved efficiency.
    /// For row-major tensors, subsequent units read lines horizontally within the tile,
    /// while for column-major tensors, they read lines vertically.
    ///
    /// # Note
    ///
    /// Out-of-bounds reads will be translated to zeros.
    pub fn load_coalesced_in_stage(
        &self,
        position: u32,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Line<EG> {
        let stage_shape_x = config.elements_in_stage_row;
        let stage_shape_y = config.elements_in_stage_col;

        let load_offsets = match config.matrix_layout {
            MatrixLayout::RowMajor => (position / stage_shape_y, position % stage_shape_y),
            MatrixLayout::ColMajor => (position % stage_shape_x, position / stage_shape_x),
        };

        self.load_coalesced(load_offsets)
    }

    fn load_coalesced(&self, load_offsets: (u32, u32)) -> Line<EG> {
        let view_x = load_offsets.0 + self.row_offset.read();
        let view_y = load_offsets.1 + self.col_offset.read();

        self.view.read_checked((self.batch_offset, view_x, view_y))
    }
}

#[cube]
/// Gives the largest slice starting at offset and not exceeding shape
fn slice_length_clamp(shape: u32, offset: u32, max_length: u32) -> u32 {
    Min::min(select(shape > offset, shape - offset, 0), max_length)
}
