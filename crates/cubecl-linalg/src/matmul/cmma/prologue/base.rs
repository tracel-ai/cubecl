use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;
use super::super::main_loop::CmmaMain;
use super::dims::{get_dims, Dimensions};
use super::offsets::{calculate_offsets, Offsets};

use crate::matmul::cmma::prologue::{make_fragments, make_shared_memories};

use super::super::prologue::{Fragments, SharedMemories};

#[cube]
pub(crate) fn prologue<D: CmmaMain, F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> (RuntimeCmmaInfo, Fragments<F, FC>, SharedMemories<FC>) {
    let runtime_info = get_runtime_info::<F, D>(lhs, rhs, out, comptime_info);
    let fragments = make_fragments::<F, FC>(comptime_info);
    let shared_memories = make_shared_memories::<FC>(comptime_info);

    (runtime_info, fragments, shared_memories)
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct RuntimeCmmaInfo {
    pub compute_ids: Ids,
    pub load_ids: Ids,
    pub dims: Dimensions,
    pub offsets: Offsets,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub plane: u32,
    pub lane: u32,
}

#[cube]
pub(crate) fn get_runtime_info<F: Float, D: CmmaMain>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> RuntimeCmmaInfo {
    let dims = get_dims::<F>(lhs, rhs);
    let offsets = calculate_offsets::<F>(lhs, rhs, out, comptime_info);

    RuntimeCmmaInfo {
        compute_ids: D::get_compute_ids(comptime_info),
        load_ids: D::get_load_ids(comptime_info),
        dims,
        offsets,
    }
}
