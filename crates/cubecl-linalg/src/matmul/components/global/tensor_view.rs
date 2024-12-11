use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global;
use crate::matmul::components::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::args::{GmmArgs, TensorInput, TensorOutput};

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorReader<GA: GmmArgs<E>, E: Numeric> {
    pub tensor: TensorInput<E, GA>,
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
pub struct TensorWriter<GA: GmmArgs<E>, E: Numeric> {
    pub tensor: TensorOutput<E, GA>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
    pub batch_offset: u32,
}

unsafe impl<GA: GmmArgs<E>, E: Numeric> Sync for TensorReader<GA, E> {}
unsafe impl<GA: GmmArgs<E>, E: Numeric> Send for TensorReader<GA, E> {}
unsafe impl<GA: GmmArgs<E>, E: Numeric> Sync for TensorWriter<GA, E> {}
unsafe impl<GA: GmmArgs<E>, E: Numeric> Send for TensorWriter<GA, E> {}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric> TensorReader<GA, EG> {
    /// Instantiate a read view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(
        tensor: TensorInput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        let rank = tensor.rank();
        let stride_x = tensor.stride(rank - 2);
        let stride_y = tensor.stride(rank - 1);
        let shape_x = tensor.shape(rank - 2);
        let shape_y = tensor.shape(rank - 1);

        TensorReader::<GA, EG> {
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
    pub fn load_coalesced<G: global::Config>(
        &self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Line<EG> {
        let line_size = config.global_line_size(ident);
        let tile_size_x = config.stage_dim(ident).tile_size_x_dim();
        let tile_size_y = config.stage_dim(ident).tile_size_y_dim();

        let view_tile_x = tile_x * tile_size_x + self.x_offset;
        let view_tile_y = tile_y * tile_size_y + self.y_offset;

        let (load_x, load_y) = match config.layout(ident) {
            MatrixLayout::RowMajor => (unit_id / tile_size_y, unit_id % tile_size_y),
            MatrixLayout::ColMajor => (unit_id % tile_size_x, unit_id / tile_size_x),
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
        self.tensor.read(position)
    }
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric> TensorWriter<GA, EG> {
    /// Instantiate a write view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(
        tensor: TensorOutput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self {
        let rank = tensor.rank();
        let stride_x = tensor.stride(rank - 2);
        let stride_y = tensor.stride(rank - 1);
        let shape_x = tensor.shape(rank - 2);
        let shape_y = tensor.shape(rank - 1);

        TensorWriter::<GA, EG> {
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

    /// Writes data into the tensor view at the specified coordinates (write_x, write_y).
    ///
    /// Each unit writes one line in a coalesced manner for improved efficiency, assuming row-major layout.
    pub fn write_coalesced<ES: Numeric, G: global::Config>(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        value: Line<ES>,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(Ident::Out);

        let view_x = tile_x * stage_dim.tile_size_x_dim()
            + unit_id / stage_dim.tile_size_y_dim()
            + self.x_offset;
        let view_y = tile_y * stage_dim.tile_size_y_dim()
            + unit_id % stage_dim.tile_size_y_dim()
            + self.y_offset;

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
