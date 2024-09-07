use crate::matmul::cmma::{base::Dimensions, config::CmmaConfig};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{BlockLoader, BlockWriter};

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockIO;

#[cube]
impl<F: Float, FC: Float> BlockLoader<F, FC> for UncheckedBlockIO {
    fn load_tile(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        batch_offset: u32,
        read_row: u32,
        read_col: u32,
        write_pos: u32,
        _dim_vertical: u32,
        dim_horizontal: u32,
    ) {
        let tensor_vec = tensor.vectorization_factor();

        let read_pos = (batch_offset + read_row * dim_horizontal + read_col) / tensor_vec;
        let value = tensor[read_pos];

        #[unroll]
        for i in 0..tensor_vec {
            shared_memory[write_pos + i] = FC::cast_from(value[i]);
        }
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for UncheckedBlockIO {
    fn write_output(
        out: &mut Tensor<F>,
        accumulator_sm: SharedMemory<F>,
        n_iter: u32,
        batch_offset: u32,
        read_position: u32,
        write_row: u32,
        write_col: u32,
        dims: Dimensions,
        #[comptime] config: CmmaConfig,
    ) {
        let tile_size = config.tile_size;
        let out_vec = out.vectorization_factor();

        let col_with_n_iter = write_col + n_iter * tile_size;

        let n_iter_read_offset = n_iter * tile_size * tile_size;
        let read_position = read_position + n_iter_read_offset;

        let write_position = batch_offset + write_row * dims.n + col_with_n_iter;

        let mut value = F::vectorized_empty(out_vec);

        #[unroll]
        for i in 0..4 {
            value[i] = accumulator_sm[read_position + i];
        }

        out[write_position / out_vec] = value;
    }
}
