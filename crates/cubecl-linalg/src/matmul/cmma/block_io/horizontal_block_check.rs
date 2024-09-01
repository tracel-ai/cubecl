use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_macros_2::{cube2, StaticExpand};

use crate::matmul::cmma::{base::Dimensions, config::CmmaConfig};

use super::base::{BlockLoader, BlockLoaderExpand, BlockWriter, BlockWriterExpand};

#[derive(StaticExpand)]
pub(crate) struct HorizontalCheckBlockIO;

#[cube2]
impl<F: Float, FC: Float> BlockLoader<F, FC> for HorizontalCheckBlockIO {
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
        let tensor_vec = vectorization(tensor);

        if read_col < dim_horizontal {
            let read_pos = (batch_offset + read_row * dim_horizontal + read_col) / tensor_vec;
            let value = tensor[read_pos];

            #[unroll]
            for i in 0..tensor_vec {
                shared_memory[write_pos + i] = FC::cast_from(value.vec_index(i));
            }
        } else {
            #[unroll]
            for i in 0..tensor_vec {
                shared_memory[write_pos + i] = FC::new(0.);
            }
        }
    }
}

#[cube2]
impl<F: Float> BlockWriter<F> for HorizontalCheckBlockIO {
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
        let out_vec = vectorization(out);

        let col_with_n_iter = write_col + n_iter * tile_size;

        if col_with_n_iter < dims.n {
            let n_iter_read_offset = n_iter * tile_size * tile_size;
            let read_position = read_position + n_iter_read_offset;

            let write_position = batch_offset + write_row * dims.n + col_with_n_iter;

            let mut value = vectorize_like(0, out);

            #[unroll]
            for i in 0..4 {
                *value.vec_index_mut(i) = accumulator_sm[read_position + i];
            }

            out[write_position / out_vec] = value;
        }
    }
}
