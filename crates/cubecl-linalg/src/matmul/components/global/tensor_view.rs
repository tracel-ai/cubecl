use crate::matmul::components::config::{self, InputIdent};
use crate::matmul::components::global;
use crate::matmul::components::{Ident, MatrixLayout};
use crate::tensor::{ReadWrite, VirtualTensor};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorReader<E: Numeric> {
    pub tensor: VirtualTensor<E>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
    pub batch_offset: u32,
}

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorWriter<E: Numeric> {
    pub tensor: VirtualTensor<E, ReadWrite>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
    pub batch_offset: u32,
}

unsafe impl<E: Numeric> Sync for TensorReader<E> {}
unsafe impl<E: Numeric> Send for TensorReader<E> {}
unsafe impl<E: Numeric> Sync for TensorWriter<E> {}
unsafe impl<E: Numeric> Send for TensorWriter<E> {}

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
            x_offset,
            y_offset,
            stride_x,
            stride_y,
            shape_x,
            shape_y,
            batch_offset,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32, #[comptime] ident: Ident) {
        match ident.as_input() {
            InputIdent::Lhs => {
                self.y_offset += k_offset;
            }
            InputIdent::Rhs => {
                self.x_offset += k_offset;
            }
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
    pub fn load_window<G: global::GlobalConfig>(
        &self,
        tile_x: u32,
        tile_y: u32,
        nth_slice: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> (Slice<Line<EG>>, u32) {
        let line_size = config.global_line_size(ident);
        let stage_tiling = config.stage_tiling(ident);
        let tile_size_x = stage_tiling.tile_shape_row();
        let tile_size_y = stage_tiling.tile_shape_col();
        let tile_lines_x = tile_size_x / line_size;
        let tile_lines_y = tile_size_y / line_size;

        let view_tile_x = tile_x * tile_size_x + self.x_offset;
        let view_tile_y = tile_y * tile_size_y + self.y_offset;

        let (load_x, load_y, num_slice_lines) = match config.layout(ident) {
            MatrixLayout::RowMajor => (nth_slice, 0, tile_lines_y),
            MatrixLayout::ColMajor => (0, nth_slice, tile_lines_x),
        };

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        let (check_x_bounds, check_y_bounds) = match ident.as_input() {
            InputIdent::Lhs => (config.check_m_bounds(), config.check_k_bounds()),
            InputIdent::Rhs => (config.check_k_bounds(), config.check_n_bounds()),
        };

        let (check_h_bounds, view_h, shape_h, check_w_bounds, view_w, shape_w) =
            match config.layout(ident) {
                MatrixLayout::RowMajor => (
                    check_x_bounds,
                    view_x,
                    self.shape_x,
                    check_y_bounds,
                    view_y,
                    self.shape_y,
                ),
                MatrixLayout::ColMajor => (
                    check_y_bounds,
                    view_y,
                    self.shape_y,
                    check_x_bounds,
                    view_x,
                    self.shape_x,
                ),
            };

        let max_slice_lines = if comptime!(check_h_bounds) {
            num_slice_lines * u32::cast_from(view_h < shape_h)
        } else {
            num_slice_lines
        };
        let size = if comptime!(check_w_bounds) {
            slice_length_clamp(shape_w / line_size, view_w / line_size, max_slice_lines)
        } else {
            max_slice_lines
        };

        // let size = match config.layout(ident) {
        //     MatrixLayout::RowMajor => match comptime!((check_x_bounds, check_y_bounds)) {
        //         (true, true) => slice_length_clamp(
        //             self.shape_y / line_size,
        //             view_y / line_size,
        //             num_slice_lines * u32::cast_from(view_x < self.shape_x),
        //         ),
        //         (true, false) => num_slice_lines * u32::cast_from(view_x < self.shape_x),
        //         (false, true) => slice_length_clamp(
        //             self.shape_y / line_size,
        //             view_y / line_size,
        //             num_slice_lines,
        //         ),
        //         (false, false) => num_slice_lines,
        //     },
        //     MatrixLayout::ColMajor => unimplemented!(),
        // };

        (self.tensor.as_slice(read_pos, read_pos + size), size)
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
    pub fn load_coalesced<G: global::GlobalConfig>(
        &self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Line<EG> {
        let line_size = config.global_line_size(ident);
        let tile_shape_x = config.stage_tiling(ident).tile_shape_row();
        let tile_shape_y = config.stage_tiling(ident).tile_shape_col();

        let view_tile_x = tile_x * tile_shape_x + self.x_offset;
        let view_tile_y = tile_y * tile_shape_y + self.y_offset;

        let (load_x, load_y) = match config.layout(ident) {
            MatrixLayout::RowMajor => (unit_id / tile_shape_y, unit_id % tile_shape_y),
            MatrixLayout::ColMajor => (unit_id % tile_shape_x, unit_id / tile_shape_x),
        };

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        let (check_x_bounds, check_y_bounds) = match ident.as_input() {
            InputIdent::Lhs => (config.check_m_bounds(), config.check_k_bounds()),
            InputIdent::Rhs => (config.check_k_bounds(), config.check_n_bounds()),
        };

        match comptime!((check_x_bounds, check_y_bounds)) {
            (true, true) => select(
                view_x < self.shape_x && view_y < self.shape_y,
                self.read(read_pos),
                Line::empty(line_size).fill(EG::from_int(0)),
            ),
            (true, false) => select(
                view_x < self.shape_x,
                self.read(read_pos),
                Line::empty(line_size).fill(EG::from_int(0)),
            ),
            (false, true) => select(
                view_y < self.shape_y,
                self.read(read_pos),
                Line::empty(line_size).fill(EG::from_int(0)),
            ),
            (false, false) => self.read(read_pos),
        }
    }

    fn read(&self, position: u32) -> Line<EG> {
        // self.tensor.as_slice(position, 1u32)[0u32]
        self.tensor.read(position)
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
    pub fn write_coalesced<ES: Numeric, G: global::GlobalConfig>(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        value: Line<ES>,
        #[comptime] config: G,
    ) {
        let tiling = config.stage_tiling(Ident::Out);

        let view_x =
            tile_x * tiling.tile_shape_row() + unit_id / tiling.tile_shape_col() + self.x_offset;
        let view_y =
            tile_y * tiling.tile_shape_col() + unit_id % tiling.tile_shape_col() + self.y_offset;

        let write_position = (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset)
            / config.global_line_size(Ident::Out);

        match comptime!((config.check_m_bounds(), config.check_n_bounds())) {
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
