use crate::components::MatmulIdent;
use crate::components::MatrixLayout;
use crate::components::global;
use cubecl_core as cubecl;
use cubecl_core::io::read_masked;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(Clone, CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorReader<EI: Numeric> {
    pub tensor: VirtualTensor<EI>,
    pub x_offset: RuntimeCell<u32>,
    pub y_offset: RuntimeCell<u32>,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
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
    pub fn new(tensor: VirtualTensor<EG>, x_offset: u32, y_offset: u32, batch_offset: u32) -> Self {
        let rank = tensor.rank();
        let stride_x = tensor.stride(rank - 2);
        let stride_y = tensor.stride(rank - 1);
        let shape_x = tensor.shape(rank - 2);
        let shape_y = tensor.shape(rank - 1);

        TensorReader::<EG> {
            tensor,
            x_offset: RuntimeCell::new(x_offset),
            y_offset: RuntimeCell::new(y_offset),
            stride_x,
            stride_y,
            shape_x,
            shape_y,
            batch_offset,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&self, k_offset: u32, #[comptime] ident: MatmulIdent) {
        match ident {
            MatmulIdent::Lhs => {
                self.y_offset.store(self.y_offset.read() + k_offset);
            }
            MatmulIdent::Rhs => {
                self.x_offset.store(self.x_offset.read() + k_offset);
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
    pub fn load_window_in_tile<G: global::GlobalConfig>(
        &self,
        tile: (u32, u32),
        nth_window: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Window<EG> {
        let line_size = config.global_line_size(ident);
        let matrix_layout = config.matrix_layout(ident);

        let tile_size_x = config.tiling_scheme().elements_in_tile_row(ident);
        let tile_size_y = config.tiling_scheme().elements_in_tile_col(ident);

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor => tile_size_y / line_size,
            MatrixLayout::ColMajor => tile_size_x / line_size,
        }};

        self.load_window::<G>(
            nth_window,
            (tile.0 * tile_size_x, tile.1 * tile_size_y),
            num_lines_in_window,
            ident,
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
    pub fn load_window_in_stage<G: global::GlobalConfig>(
        &self,
        nth_window: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Window<EG> {
        let line_size = config.global_line_size(ident);
        let matrix_layout = config.matrix_layout(ident);

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor =>
                config.tiling_scheme().elements_in_stage_col(ident) / line_size
            ,
            MatrixLayout::ColMajor =>
                config.tiling_scheme().elements_in_stage_row(ident) / line_size
            ,
        }};

        self.load_window::<G>(
            nth_window,
            (0u32, 0u32).runtime(),
            num_lines_in_window,
            ident,
            config,
        )
    }

    fn load_window<G: global::GlobalConfig>(
        &self,
        nth_window: u32,
        tile_offsets: (u32, u32),
        #[comptime] num_lines_in_window: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Window<EG> {
        let line_size = config.global_line_size(ident);
        let matrix_layout = config.matrix_layout(ident);

        let (load_x, load_y) = match matrix_layout {
            MatrixLayout::RowMajor => (nth_window, 0),
            MatrixLayout::ColMajor => (0, nth_window),
        };

        let view_tile_x = tile_offsets.0 + self.x_offset.read();
        let view_tile_y = tile_offsets.1 + self.y_offset.read();

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        let (check_h_bounds, view_h, shape_h, check_w_bounds, view_w, shape_w) =
            match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => (
                    config.check_row_bounds(ident),
                    view_x,
                    self.shape_x,
                    config.check_col_bounds(ident),
                    view_y,
                    self.shape_y,
                ),
                MatrixLayout::ColMajor => (
                    config.check_col_bounds(ident),
                    view_y,
                    self.shape_y,
                    config.check_row_bounds(ident),
                    view_x,
                    self.shape_x,
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
            slice: self.tensor.as_slice(read_pos, read_pos + size),
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
    pub fn load_coalesced_in_tile<G: global::GlobalConfig>(
        &self,
        tile_x: u32,
        tile_y: u32,
        position: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let tile_size_x = config.tiling_scheme().elements_in_tile_row(ident);
        let tile_size_y = config.tiling_scheme().elements_in_tile_col(ident);

        let view_tile_x = tile_x * tile_size_x;
        let view_tile_y = tile_y * tile_size_y;

        let (load_x, load_y) = match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => (position / tile_size_y, position % tile_size_y),
            MatrixLayout::ColMajor => (position % tile_size_x, position / tile_size_x),
        };

        self.load_coalesced::<G>((load_x + view_tile_x, load_y + view_tile_y), ident, config)
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
    pub fn load_coalesced_in_stage<G: global::GlobalConfig>(
        &self,
        position: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let stage_shape_x = config.tiling_scheme().elements_in_stage_row(ident);
        let stage_shape_y = config.tiling_scheme().elements_in_stage_col(ident);

        let load_offsets = match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => (position / stage_shape_y, position % stage_shape_y),
            MatrixLayout::ColMajor => (position % stage_shape_x, position / stage_shape_x),
        };

        self.load_coalesced::<G>(load_offsets, ident, config)
    }

    fn load_coalesced<G: global::GlobalConfig>(
        &self,
        load_offsets: (u32, u32),
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let line_size = config.global_line_size(ident);

        let view_x = load_offsets.0 + self.x_offset.read();
        let view_y = load_offsets.1 + self.y_offset.read();

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        match comptime!((
            config.check_row_bounds(ident),
            config.check_col_bounds(ident)
        )) {
            (true, true) => read_masked::<Line<EG>>(
                view_x < self.shape_x && view_y < self.shape_y,
                self.tensor.as_slice(0, self.tensor.len()),
                read_pos,
                Line::cast_from(0),
            ),
            (true, false) => read_masked::<Line<EG>>(
                view_x < self.shape_x,
                self.tensor.as_slice(0, self.tensor.len()),
                read_pos,
                Line::cast_from(0),
            ),
            (false, true) => read_masked::<Line<EG>>(
                view_y < self.shape_y,
                self.tensor.as_slice(0, self.tensor.len()),
                read_pos,
                Line::cast_from(0),
            ),
            (false, false) => self.tensor.read(read_pos),
        }
    }
}

#[cube]
/// Gives the largest slice starting at offset and not exceeding shape
fn slice_length_clamp(shape: u32, offset: u32, max_length: u32) -> u32 {
    Min::min(select(shape > offset, shape - offset, 0), max_length)
}
