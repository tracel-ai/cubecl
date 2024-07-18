use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::Dimensions;
use crate::matmul::cmma::config::CmmaConfig;

#[cube]
pub(crate) trait BlockLoader<F: Float, FC: Float>: Send + Sync + 'static {
    fn load_tile(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        batch_offset: UInt,
        read_row: UInt,
        read_col: UInt,
        write_pos: UInt,
        dim_vertical: UInt,
        dim_horizontal: UInt,
    );
}

#[cube]
pub(crate) trait BlockWriter<F: Float>: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
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
    );
}
