use crate::components::{MatmulIdent, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

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

unsafe impl<EG: Numeric> Sync for TensorWriter<EG> {}
unsafe impl<EG: Numeric> Send for TensorWriter<EG> {}

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
        let tile_size_m = config.tiling_scheme().elements_in_tile_m();
        let tile_size_n = config.tiling_scheme().elements_in_tile_n();

        let view_x = tile_x * tile_size_m + unit_id / tile_size_n + self.x_offset;
        let view_y = tile_y * tile_size_n + unit_id % tile_size_n + self.y_offset;

        let write_position = (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset)
            / config.global_line_size(MatmulIdent::Out);

        match comptime!((
            config.check_row_bounds(MatmulIdent::Out),
            config.check_col_bounds(MatmulIdent::Out)
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
