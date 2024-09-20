use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{
    base::{Fragments, Ids, SharedMemories},
    compute_loop::base::{get_smem_position_rhs, load_into_fragment},
    config::ComptimeCmmaInfo,
};

use super::base::{get_smem_position_lhs, ComputeLoop};

pub(crate) struct AllAccumulatorsFirstComputeLoop {}

#[cube]
impl ComputeLoop for AllAccumulatorsFirstComputeLoop {
    fn compute_loop<F: Float, FC: Float>(
        shared_memories: SharedMemories<FC>,
        fragments: &mut Fragments<F, FC>,
        ids: Ids,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        // Comptime values
        let block_size_k = comptime_info.block_size_k;
        let block_size_n = comptime_info.block_size_n;
        let tile_size = comptime_info.tile_size;
        let unroll = comptime_info.unroll;
        let num_accumulators = comptime_info.num_accumulators;
        let num_buffers = block_size_k / tile_size;
        let smem_h = block_size_n / tile_size;
        let num_coop_per_row = smem_h / num_accumulators;

        // Runtime values
        let tile_row = ids.coop / num_coop_per_row;
        let tile_col_base = (ids.coop % num_coop_per_row) * num_accumulators;

        #[unroll(unroll)]
        for buffer_iter in 0..num_buffers {
            #[unroll]
            for accumulator_iter in 0..num_accumulators {
                load_into_fragment(
                    get_smem_position_lhs::<F, FC>(tile_row, buffer_iter, comptime_info),
                    shared_memories.lhs,
                    &fragments.lhs,
                    comptime_info,
                );

                load_into_fragment(
                    get_smem_position_rhs::<F, FC>(
                        buffer_iter,
                        tile_col_base + accumulator_iter,
                        comptime_info,
                    ),
                    shared_memories.rhs,
                    &fragments.rhs,
                    comptime_info,
                );

                let accumulator = &fragments.accumulators.index(accumulator_iter);
                cmma::execute::<FC, FC, F, F>(
                    &fragments.lhs,
                    &fragments.rhs,
                    accumulator,
                    accumulator,
                );
            }
        }
    }
}
