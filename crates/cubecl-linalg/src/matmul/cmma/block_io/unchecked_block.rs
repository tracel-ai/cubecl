use crate::matmul::cmma::load_shared_memory::load_info::LoadInfo;
use crate::matmul::cmma::prologue::{Dimensions, RuntimeCmmaInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{BlockLoader, BlockWriter};

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockIO;

#[cube]
impl<F: Float, FC: Float> BlockLoader<F, FC> for UncheckedBlockIO {
    fn load_single<I: LoadInfo>(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        read_row: u32,
        read_col: u32,
        write_pos: u32,
        runtime_info: RuntimeCmmaInfo,
    ) {
        let tensor_vec = vectorization_of(tensor);
        let is_scalar = tensor_vec == 1;

        let read_pos =
            (I::batch_offset(runtime_info) + read_row * I::dim_horizontal(runtime_info) + read_col)
                / tensor_vec;
        let value = tensor[read_pos];

        if is_scalar {
            shared_memory[write_pos] = FC::cast_from(value);
        } else {
            #[unroll]
            for i in 0..tensor_vec {
                shared_memory[write_pos + i] = FC::cast_from(value[i]);
            }
        }
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for UncheckedBlockIO {
    fn write_single(
        out: &mut Tensor<F>,
        accumulator_sm: SharedMemory<F>,
        batch_offset: u32,
        read_position: u32,
        write_row: u32,
        write_col: u32,
        dims: Dimensions,
    ) {
        let out_vec = vectorization_of(out);
        let is_scalar = out_vec == 1;

        let write_position = batch_offset + write_row * dims.n + write_col;

        if is_scalar {
            let val = accumulator_sm[read_position];
            out[write_position / out_vec] = val;
        } else {
            let mut value = F::vectorized_empty(out_vec);

            #[unroll]
            for i in 0..out_vec {
                value[i] = accumulator_sm[read_position + i];
            }

            out[write_position / out_vec] = value;
        }
    }
}
