use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::{BlockLoader, BlockWriter};

use crate::matmul::kernels::cmma_old::load_shared_memory::load_info::LoadInfo;
use crate::matmul::kernels::cmma_old::prologue::{Dimensions, RuntimeCmmaInfo};

pub(crate) struct HorizontalCheckBlockIO;

#[cube]
impl<F: Float, FC: Float> BlockLoader<F, FC> for HorizontalCheckBlockIO {
    fn load_single<I: LoadInfo>(
        tensor: &Tensor<Line<F>>,
        shared_memory: &mut SharedMemory<Line<FC>>,
        read_row: u32,
        read_col: u32,
        write_pos: u32,
        runtime_info: RuntimeCmmaInfo,
    ) {
        let tensor_vec = tensor.line_size();
        let is_scalar = tensor_vec == 1;
        let dim_horizontal = I::dim_horizontal(runtime_info); // = gmem_stride

        if read_col < dim_horizontal {
            let read_pos =
                (I::batch_offset(runtime_info) + read_row * dim_horizontal + read_col) / tensor_vec;
            let value = tensor[read_pos];

            if is_scalar {
                shared_memory[write_pos] = FC::cast_from(value);
            } else {
                #[unroll]
                for i in 0..tensor_vec {
                    shared_memory[write_pos + i] = FC::cast_from(value[i]);
                }
            }
        } else {
            #[unroll]
            for i in 0..tensor_vec {
                shared_memory[write_pos + i] = FC::new(0.);
            }
        }
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for HorizontalCheckBlockIO {
    fn write_single(
        out: &mut Tensor<Line<F>>,
        accumulator_sm: SharedMemory<Line<F>>,
        batch_offset: u32,
        read_position: u32,
        write_row: u32,
        write_col: u32,
        dims: Dimensions,
    ) {
        let out_vec = out.line_size();
        let is_scalar = out_vec == 1;

        if write_col < dims.n {
            let write_position = batch_offset + write_row * dims.n + write_col;
            let val = accumulator_sm[read_position];
        }
    }
}
