use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::prologue::Ids;

use super::super::{
    compute_loop::base::compute_loop,
    config::ComptimeCmmaInfo,
    epilogue::base::write_to_output,
    load_shared_memory::base::load_to_shared_memories,
    prologue::{Fragments, RuntimeCmmaInfo, SharedMemories},
};
use super::base::MainLoop;

pub(crate) struct StandardMainLoop {}

#[cube]
impl MainLoop for StandardMainLoop {
    fn main_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        shared_memories: SharedMemories<FC>,
        fragments: &mut Fragments<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let b_k = comptime_info.block_size_k;
        let num_loops = (runtime_info.dims.k + b_k - 1) / b_k;

        for block in 0..num_loops {
            let k_offset = block * comptime_info.block_size_k;

            load_to_shared_memories::<F, FC>(
                lhs,
                rhs,
                shared_memories,
                k_offset,
                runtime_info,
                comptime_info,
            );

            sync_units();

            compute_loop::<F, FC>(
                shared_memories,
                fragments,
                runtime_info.compute_ids,
                comptime_info,
            );

            sync_units();
        }
    }

    fn epilogue<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        write_to_output(out, accumulators, runtime_info, comptime_info);
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }
}
