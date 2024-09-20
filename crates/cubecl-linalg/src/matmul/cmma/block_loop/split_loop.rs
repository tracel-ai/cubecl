use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::runtime_info::Ids;

use super::super::{
    compute_loop::base::compute_loop,
    config::ComptimeCmmaInfo,
    load_shared_memory::base::load_to_shared_memories,
    runtime_info::{Fragments, RuntimeCmmaInfo, SharedMemories},
    write_output::base::write_to_output,
};
use super::base::BlockLoop;

/// Assumes CUBE_DIM_Y = comptime_info.num_compute_coops + comptime_info.num_load_coops
pub(crate) struct SplitBlockLoop {}

#[cube]
impl BlockLoop for SplitBlockLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        shared_memories: SharedMemories<FC>,
        mut fragments: Fragments<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let b_k = comptime_info.block_size_k;
        let num_loops = (runtime_info.dims.k + b_k - 1) / b_k;

        for block in 0..num_loops {
            let k_offset = block * comptime_info.block_size_k;

            if !is_compute_coop(comptime_info) {
                load_to_shared_memories::<F, FC>(
                    lhs,
                    rhs,
                    shared_memories,
                    k_offset,
                    runtime_info,
                    comptime_info,
                );
            }

            sync_units();

            if is_compute_coop(comptime_info) {
                compute_loop::<F, FC>(
                    shared_memories,
                    &mut fragments,
                    runtime_info.compute_ids,
                    comptime_info,
                );
            }

            sync_units();
        }

        if is_compute_coop(comptime_info) {
            write_to_output(out, fragments.accumulators, runtime_info, comptime_info);
        }
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y / 2,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y / 2,
            lane: UNIT_POS_X,
        }
    }
}

#[cube]
fn is_compute_coop(#[comptime] comptime_info: ComptimeCmmaInfo) -> bool {
    UNIT_POS_Y < comptime_info.num_compute_coops
}
