use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{base::Dimensions, config::CmmaConfig};

use super::base::{BlockLoader, BlockWriter};

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockIO;

#[cube]
impl<F: Float, FC: Float> BlockLoader<F, FC> for UncheckedBlockIO {
    fn load_tile(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        batch_offset: UInt,
        read_row: UInt,
        read_col: UInt,
        write_pos: UInt,
        _dim_vertical: UInt,
        dim_horizontal: UInt,
    ) {
        let tensor_vec = Comptime::vectorization(tensor);
        let tensor_vec_r = Comptime::runtime(tensor_vec);

        let read_pos = (batch_offset + read_row * dim_horizontal + read_col) / tensor_vec_r;
        let value = tensor[read_pos];

        for i in range(0u32, Comptime::get(tensor_vec), Comptime::new(true)) {
            shared_memory[write_pos + i] = FC::cast_from(value[i]);
        }
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for UncheckedBlockIO {
    fn write_output(
        out: &mut Tensor<F>,
        accumulator_sm: SharedMemory<F>,
        n_iter: UInt,
        batch_offset: UInt,
        read_position: UInt,
        write_row: UInt,
        write_col: UInt,
        dims: Dimensions,
        config: Comptime<CmmaConfig>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let out_vec = Comptime::vectorization(out);
        let out_vec_r = Comptime::runtime(out_vec);

        let col_with_n_iter = write_col + n_iter * Comptime::runtime(tile_size);

        let n_iter_read_offset = n_iter * Comptime::runtime(tile_size * tile_size);
        let read_position = read_position + n_iter_read_offset;

        let write_position = batch_offset + write_row * dims.n + col_with_n_iter;

        let mut value = F::vectorized_empty(Comptime::get(out_vec));

        for i in range(0u32, 4u32, Comptime::new(true)) {
            value[i] = accumulator_sm[read_position + i];
        }

        out[write_position / out_vec_r] = value;
    }
}
