use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct TensorView<E: Numeric> {
    pub tensor: Tensor<Line<E>>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_x: u32,
    pub stride_y: u32,
    pub shape_x: u32,
    pub shape_y: u32,
    pub batch_offset: u32,
}

#[cube]
impl<EG: Numeric> TensorView<EG> {
    pub fn new(
        tensor: Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
    ) -> TensorView<EG> {
        let rank = tensor.rank();
        let stride_b = tensor.stride(rank - 3);
        let stride_x = tensor.stride(rank - 2);
        let stride_y = tensor.stride(rank - 1);
        let shape_x = tensor.shape(rank - 2);
        let shape_y = tensor.shape(rank - 1);
        let batch_offset = stride_b * nth_batch;

        TensorView::<EG> {
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

    pub fn update_view(&mut self, k_offset: u32, #[comptime] ident: Ident) {
        match ident {
            Ident::Lhs => {
                self.y_offset += k_offset;
            }
            Ident::Rhs => {
                self.x_offset += k_offset;
            }
            Ident::Out => {}
        }
    }

    pub fn load_coalesced<G: GmmConfig>(
        &self,
        tile_x: u32,
        tile_y: u32,
        load_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Line<EG> {
        let tensor = &self.tensor;
        let line_size = config.line_size(ident);
        let tile_size_x = config.stage_dim(ident).tile_size_x;
        let tile_size_y = config.stage_dim(ident).tile_size_y;

        let view_tile_x = tile_x * tile_size_x + self.x_offset;
        let view_tile_y = tile_y * tile_size_y + self.y_offset;

        let (load_x, load_y) = match config.layout(ident) {
            MatrixLayout::RowMajor => (load_id / tile_size_y, load_id % tile_size_y),
            MatrixLayout::ColMajor => (load_id % tile_size_x, load_id / tile_size_x),
        };

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let read_pos =
            (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset) / line_size;

        select(
            view_x < self.shape_x && view_y < self.shape_y,
            tensor[read_pos],
            Line::empty(line_size).fill(EG::from_int(0)),
        )
    }

    pub fn write_coalesced<ES: Numeric, G: GmmConfig>(
        &mut self,
        write_x: u32,
        write_y: u32,
        value: Line<ES>,
        #[comptime] config: G,
    ) {
        let tensor = &mut self.tensor;
        let view_x = write_x + self.x_offset;
        let view_y = write_y + self.y_offset;

        let write_position = (view_x * self.stride_x + view_y * self.stride_y + self.batch_offset)
            / tensor.line_size();

        if config.check_m_bounds() {
            if config.check_n_bounds() {
                if write_x < self.shape_x && write_y < self.shape_y {
                    tensor[write_position] = Line::cast_from(value);
                }
            } else {
                if write_x < self.shape_x {
                    tensor[write_position] = Line::cast_from(value);
                }
            }
        } else {
            if config.check_n_bounds() {
                if write_y < self.shape_y {
                    tensor[write_position] = Line::cast_from(value);
                }
            } else {
                tensor[write_position] = Line::cast_from(value);
            }
        }
    }
}
