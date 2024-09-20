use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use crate::matmul::cmma::runtime_info::ids::IdDispatch;

use super::super::config::ComptimeCmmaInfo;
use super::dims::{get_dims, Dimensions};
use super::ids::{Ids, SameRoleIdDispatch};
use super::offsets::{calculate_offsets, Offsets};

#[derive(CubeType, Copy, Clone)]
pub(crate) struct RuntimeCmmaInfo {
    pub compute_ids: Ids,
    pub load_ids: Ids,
    pub dims: Dimensions,
    pub offsets: Offsets,
}

#[cube]
pub(crate) fn get_runtime_info<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> RuntimeCmmaInfo {
    let dims = get_dims::<F>(lhs, rhs);
    let offsets = calculate_offsets::<F>(lhs, rhs, out, comptime_info);

    RuntimeCmmaInfo {
        compute_ids: SameRoleIdDispatch::get_compute_ids(comptime_info),
        load_ids: SameRoleIdDispatch::get_load_ids(comptime_info),
        dims,
        offsets,
    }
}
