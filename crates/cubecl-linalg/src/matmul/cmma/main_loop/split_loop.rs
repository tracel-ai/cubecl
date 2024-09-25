use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::compute_loop::base::ComputeLoop;
use crate::matmul::cmma::prologue::{prologue, Ids};

use super::super::{
    config::ComptimeCmmaInfo,
    epilogue::base::write_to_output,
    load_shared_memory::base::load_to_shared_memories,
    prologue::{Fragments, RuntimeCmmaInfo, SharedMemories},
};
use super::base::CmmaMain;

pub(crate) struct SplitMainLoop {}

#[cube]
impl CmmaMain for SplitMainLoop {
    fn prologue<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) -> (RuntimeCmmaInfo, Fragments<F, FC>, SharedMemories<FC>) {
        prologue::<SplitMainLoop, F, FC>(lhs, rhs, out, comptime_info)
    }

    fn main_loop<C: ComputeLoop, F: Float, FC: Float>(
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

            if !is_compute_plane(comptime_info) {
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

            if is_compute_plane(comptime_info) {
                C::compute_loop::<F, FC>(
                    shared_memories,
                    fragments,
                    runtime_info.compute_ids,
                    comptime_info,
                );
            }

            sync_units();
        }
    }

    fn epilogue<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        if is_compute_plane(comptime_info) {
            write_to_output(out, accumulators, runtime_info, comptime_info);
        }
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            plane: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids {
        Ids {
            plane: UNIT_POS_Y - comptime_info.num_compute_planes,
            lane: UNIT_POS_X,
        }
    }
}

#[cube]
fn is_compute_plane(#[comptime] comptime_info: ComptimeCmmaInfo) -> bool {
    UNIT_POS_Y < comptime_info.num_compute_planes
}
