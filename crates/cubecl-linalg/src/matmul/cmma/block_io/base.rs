use crate::matmul::cmma::base::Dimensions;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub(crate) trait BlockLoader<F: Float, FC: Float> {
    fn load_tile(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        batch_offset: u32,
        read_row: u32,
        read_col: u32,
        write_pos: u32,
        dim_vertical: u32,
        dim_horizontal: u32,
    );
}

#[cube]
pub(crate) trait BlockWriter<F: Float>: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn write_output(
        out: &mut Tensor<F>,
        accumulator_sm: SharedMemory<F>,
        batch_offset: u32,
        read_position: u32,
        write_row: u32,
        write_col: u32,
        dims: Dimensions,
    );
}
