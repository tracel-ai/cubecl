use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::runtime_info::Ids;

use super::super::{
    config::ComptimeCmmaInfo,
    runtime_info::{Fragments, RuntimeCmmaInfo, SharedMemories},
};
use super::base::BlockLoop;

pub(crate) struct SplitBlockLoop {}

#[cube]
/// Assumes CUBE_DIM_Y = comptime_info.num_compute_coops + comptime_info.num_load_coops
impl BlockLoop for SplitBlockLoop {
    fn block_loop<F: Float, FC: Float>(
        _lhs: &Tensor<F>,
        _rhs: &Tensor<F>,
        _out: &mut Tensor<F>,
        _shared_memories: SharedMemories<FC>,
        _fragments: Fragments<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let b_k = comptime_info.block_size_k;
        let _num_loops = (runtime_info.dims.k + b_k - 1) / b_k;

        // TODO
    }

    fn get_compute_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        // TODO shouldn't even have id if not a compute coop
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }

    fn get_load_ids(#[comptime] _comptime_info: ComptimeCmmaInfo) -> Ids {
        // TODO shouldn't even have id if not a load coop
        Ids {
            coop: UNIT_POS_Y,
            lane: UNIT_POS_X,
        }
    }
}
