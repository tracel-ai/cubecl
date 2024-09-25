use crate::matmul::cmma::load_shared_memory::load_info::LoadInfo;
use crate::matmul::cmma::prologue::{Dimensions, RuntimeCmmaInfo};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub(crate) trait BlockLoader<F: Float, FC: Float> {
    fn load_single<I: LoadInfo>(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<FC>,
        read_row: u32,
        read_col: u32,
        write_pos: u32,
        runtime_info: RuntimeCmmaInfo,
    );
}

#[cube]
pub(crate) trait BlockWriter<F: Float>: Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn write_single(
        out: &mut Tensor<F>,
        accumulator_sm: SharedMemory<F>,
        batch_offset: u32,
        read_position: u32,
        write_row: u32,
        write_col: u32,
        dims: Dimensions,
    );
}
