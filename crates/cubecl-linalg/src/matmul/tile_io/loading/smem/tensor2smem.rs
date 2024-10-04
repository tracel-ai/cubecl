use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;

#[cube]
pub(crate) fn tensor_to_shared_memory<E: Numeric>(
    gmem: &Tensor<Line<E>>,
    smem: &mut SharedMemory<Line<E>>,
    gmem_x_offset: u32,
    gmem_y_offset: u32,
    block_info: BlockInfo,
) {
}
