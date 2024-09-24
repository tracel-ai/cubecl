use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;
use super::super::main_loop::MainLoop;
use super::dims::{get_dims, Dimensions};
use super::offsets::{calculate_offsets, Offsets};

#[derive(CubeType, Copy, Clone)]
pub(crate) struct RuntimeCmmaInfo {
    pub compute_ids: Ids,
    pub load_ids: Ids,
    pub dims: Dimensions,
    pub offsets: Offsets,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: u32,
    pub lane: u32,
}

#[cube]
pub(crate) fn get_runtime_info<F: Float, D: MainLoop>(
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
