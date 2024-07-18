use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::config::CmmaConfig;

use super::base::{BlockLoader, BlockWriter, CheckBounds, ReadTileInfo, WriteTileInfo};

pub(crate) struct HorizontalCheckBlockIO;

#[cube]
impl<F: Float, FC: Float> BlockLoader<F, FC> for HorizontalCheckBlockIO {
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

        if read_col < dim_horizontal {
            let read_pos = batch_offset + (read_row * dim_horizontal + read_col) / tensor_vec_r;
            let value = tensor[read_pos];

            for i in range(0u32, Comptime::get(tensor_vec), Comptime::new(true)) {
                shared_memory[write_pos + i] = FC::cast_from(value[i]);
            }
        } else {
            for i in range(0u32, Comptime::get(tensor_vec), Comptime::new(true)) {
                shared_memory[write_pos + i] = FC::new(0.);
            }
        }
    }
}

#[cube]
impl<F: Float> BlockWriter<F> for HorizontalCheckBlockIO {
    fn write_output(
        out: &mut Tensor<F>,
        results: &Array<F>,
        info: WriteTileInfo,
        config: Comptime<CmmaConfig>,
        check_bounds: CheckBounds,
    ) {
    }
}
