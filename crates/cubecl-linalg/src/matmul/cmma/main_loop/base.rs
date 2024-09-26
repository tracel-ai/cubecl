use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::compute_loop::base::ComputeLoop;
use crate::matmul::cmma::prologue::Ids;

use super::super::{
    config::ComptimeCmmaInfo,
    prologue::{Fragments, RuntimeCmmaInfo, SharedMemories},
};

#[cube]
pub(crate) trait CmmaMain {
    fn prologue<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) -> (RuntimeCmmaInfo, Fragments<F, FC>, SharedMemories<FC>);

    fn main_loop<C: ComputeLoop, F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        shared_memories: SharedMemories<FC>,
        fragments: &mut Fragments<F, FC>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );

    fn epilogue<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );

    fn get_compute_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;

    fn get_load_ids(#[comptime] comptime_info: ComptimeCmmaInfo) -> Ids;
}
