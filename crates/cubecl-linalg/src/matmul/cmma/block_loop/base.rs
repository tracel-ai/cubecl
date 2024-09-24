use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::runtime_info::Ids;

use super::super::{
    config::ComptimeCmmaInfo,
    runtime_info::{Fragments, RuntimeCmmaInfo, SharedMemories},
};

#[cube]
pub(crate) trait BlockLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        shared_memories: SharedMemories<FC>,
        fragments: Fragments<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
    fn get_compute_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;
    fn get_load_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;
}
