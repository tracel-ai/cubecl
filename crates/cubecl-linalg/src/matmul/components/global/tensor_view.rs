use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global;
use crate::matmul::components::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::io::read_masked;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

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

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Uses a [`TensorMap`] to actually execute the load.
pub struct MappedTensorReader<EG: Numeric> {
    pub tensor: TensorMap<EG>,
    pub tile_x: u32,
    pub tile_y: u32,
    pub batch: u32,
}

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorWriter<EO: Numeric> {
    pub tensor: VirtualTensor<EO, ReadWrite>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
    pub batch_offset: u32,
}

unsafe impl<EG: Numeric> Sync for TensorReader<EG> {}
unsafe impl<EG: Numeric> Send for TensorReader<EG> {}
unsafe impl<EG: Numeric> Sync for MappedTensorReader<EG> {}
unsafe impl<EG: Numeric> Send for MappedTensorReader<EG> {}
unsafe impl<EG: Numeric> Sync for TensorWriter<EG> {}
unsafe impl<EG: Numeric> Send for TensorWriter<EG> {}

#[derive(CubeType)]
/// Contiguous slice wrapper for memcpy_async loading
pub struct Window<EG: Numeric> {
    /// Contiguous slice containing all and only data of window
    pub slice: Slice<Line<EG>>,
    /// Number of lines
    pub size: u32,
}

#[cube]
impl<EG: Numeric> MappedTensorReader<EG> {
    /// Instantiate a read view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(tensor: TensorMap<EG>, tile_x: u32, tile_y: u32, batch: u32) -> Self {
        MappedTensorReader::<EG> {
            tensor,
            tile_x,
            tile_y,
            batch,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32, #[comptime] ident: Ident) {
        match ident.as_input_ident() {
            InputIdent::Lhs => {
                self.tile_y += k_offset;
            }
            InputIdent::Rhs => {
                self.tile_x += k_offset;
            }
        }
    }
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
    pub fn update_view(&self, k_offset: u32, #[comptime] ident: InputIdent) {
        match ident {
            InputIdent::Lhs => {
                self.y_offset.store(self.y_offset.read() + k_offset);
            }
            InputIdent::Rhs => {
                self.x_offset.store(self.x_offset.read() + k_offset);
            }
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
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Window<EG> {
        let line_size = config.global_line_size(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let matrix_layout = config.matrix_layout(input_ident);

        let tile_size_x = tiling_dimensions.tile_shape_row();
        let tile_size_y = tiling_dimensions.tile_shape_col();

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor => tile_size_y / line_size,
            MatrixLayout::ColMajor => tile_size_x / line_size,
        }};

        self.load_window::<G>(
            nth_window,
            (tile.0 * tile_size_x, tile.1 * tile_size_y),
            num_lines_in_window,
            input_ident,
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
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Window<EG> {
        let line_size = config.global_line_size(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let matrix_layout = config.matrix_layout(input_ident);

        let num_lines_in_window = comptime! {match matrix_layout {
            MatrixLayout::RowMajor =>
                tiling_dimensions.total_col() / line_size
            ,
            MatrixLayout::ColMajor =>
                tiling_dimensions.total_row() / line_size
            ,
        }};

        self.load_window::<G>(
            nth_window,
            (0u32, 0u32).runtime(),
            num_lines_in_window,
            input_ident,
            config,
        )
    }

    fn load_window<G: global::GlobalConfig>(
        &self,
        nth_window: u32,
        tile_offsets: (u32, u32),
        #[comptime] num_lines_in_window: u32,
        #[comptime] ident: InputIdent,
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
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let tile_shape_x = config.tiling_dimensions(input_ident).tile_shape_row();
        let tile_shape_y = config.tiling_dimensions(input_ident).tile_shape_col();

        let view_tile_x = tile_x * tile_shape_x;
        let view_tile_y = tile_y * tile_shape_y;

        let (load_x, load_y) = match config.matrix_layout(input_ident) {
            MatrixLayout::RowMajor => (position / tile_shape_y, position % tile_shape_y),
            MatrixLayout::ColMajor => (position % tile_shape_x, position / tile_shape_x),
        };

        self.load_coalesced::<G>(
            (load_x + view_tile_x, load_y + view_tile_y),
            input_ident,
            config,
        )
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
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let stage_shape_x = config.tiling_dimensions(input_ident).total_row();
        let stage_shape_y = config.tiling_dimensions(input_ident).total_col();

        let load_offsets = match config.matrix_layout(input_ident) {
            MatrixLayout::RowMajor => (position / stage_shape_y, position % stage_shape_y),
            MatrixLayout::ColMajor => (position % stage_shape_x, position / stage_shape_x),
        };

        self.load_coalesced::<G>(load_offsets, input_ident, config)
    }

    fn load_coalesced<G: global::GlobalConfig>(
        &self,
        load_offsets: (u32, u32),
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Line<EG> {
        let line_size = config.global_line_size(input_ident);

        let view_x = load_offsets.0 + self.x_offset.read();
        let view_y = load_offsets.1 + self.y_offset.read();

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        match comptime!((
            config.check_row_bounds(input_ident),
            config.check_col_bounds(input_ident)
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
impl<EG: Numeric> TensorWriter<EG> {
    /// Instantiate a write view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(
        tensor: VirtualTensor<EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        let rank = tensor.rank();
        let stride_x = tensor.stride(rank - 2);
        let stride_y = tensor.stride(rank - 1);
        let shape_x = tensor.shape(rank - 2);
        let shape_y = tensor.shape(rank - 1);

        TensorWriter::<EG> {
            tensor,
            x_offset,
            y_offset,
            stride_x,
            stride_y,
            shape_x,
            shape_y,
            batch_offset,
        }
    }

    /// Writes data into the tensor view at the specified coordinates (tile_x, tile_y).
    ///
    /// Each unit writes one line in a coalesced manner for improved efficiency, assuming row-major layout.
    pub fn write_coalesced<G: global::GlobalConfig>(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        value: Line<EG>,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(Ident::Out);

        let view_x =
            tile_x * tiling.tile_shape_row() + unit_id / tiling.tile_shape_col() + self.x_offset;
        let view_y =
            tile_y * tiling.tile_shape_col() + unit_id % tiling.tile_shape_col() + self.y_offset;

        let write_position = (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset)
            / config.global_line_size(Ident::Out);

        match comptime!((
            config.check_row_bounds(Ident::Out),
            config.check_col_bounds(Ident::Out)
        )) {
            (true, true) => {
                if view_x < self.shape_x && view_y < self.shape_y {
                    self.write(write_position, Line::cast_from(value));
                }
            }
            (true, false) => {
                if view_x < self.shape_x {
                    self.write(write_position, Line::cast_from(value));
                }
            }
            (false, true) => {
                if view_y < self.shape_y {
                    self.write(write_position, Line::cast_from(value));
                }
            }
            (false, false) => {
                self.write(write_position, Line::cast_from(value));
            }
        }
    }

    fn write(&mut self, position: u32, value: Line<EG>) {
        self.tensor.write(position, value)
    }
}

#[cube]
/// Gives the largest slice starting at offset and not exceeding shape
fn slice_length_clamp(shape: u32, offset: u32, max_length: u32) -> u32 {
    Min::min(select(shape > offset, shape - offset, 0), max_length)
}
