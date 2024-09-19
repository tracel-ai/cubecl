use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{
    base::{Fragments, Ids, SharedMemories},
    compute_loop::{
        accumulators_first::AllAccumulatorsFirstComputeLoop,
        buffers_first::AllBuffersFirstComputeLoop,
    },
    config::ComptimeCmmaInfo,
};

#[cube]
pub(crate) fn compute_loop<F: Float, FC: Float>(
    shared_memories: SharedMemories<FC>,
    fragments: &mut Fragments<F, FC>,
    ids: Ids,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    if comptime_info.compute_loop_order_strategy == 0 {
        AllBuffersFirstComputeLoop::compute_loop(shared_memories, fragments, ids, comptime_info);
    } else {
        AllAccumulatorsFirstComputeLoop::compute_loop(
            shared_memories,
            fragments,
            ids,
            comptime_info,
        );
    }
}

#[cube]
pub(crate) trait ComputeLoop {
    fn compute_loop<F: Float, FC: Float>(
        shared_memories: SharedMemories<FC>,
        fragments: &mut Fragments<F, FC>,
        ids: Ids,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

#[cube]
pub(crate) fn load_into_fragment<FC: Float>(
    tile: u32,
    smem: SharedMemory<FC>,
    fragment: &cmma::Matrix<FC>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let tile_size = comptime_info.tile_size;
    let smem_stride = tile_size * tile_size;

    let smem_pos = tile * smem_stride;
    let slice = smem.slice(smem_pos, smem_pos + smem_stride);
    cmma::load::<FC>(fragment, slice, 16);
}
