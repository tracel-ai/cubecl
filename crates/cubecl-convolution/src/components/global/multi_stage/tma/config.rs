use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{LhsS, MatmulPrecision, RhsS, TilingScheme};

use crate::components::ConvolutionProblem;

/// More than 4 stages would likely slow things down from code size
/// Should test more to find the ideal value here, just using 4 because that's what cuDNN uses
const NUM_STAGES_MAX: u32 = 8;
/// I found that too many pipeline stages relative to k degrade performance
const MIN_STAGES_PER_PIPELINE: u32 = 32;

pub(crate) fn num_stages<R: Runtime, MP: MatmulPrecision>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &ConvolutionProblem,
    num_planes: u32,
    tiling_scheme: &TilingScheme,
) -> u32 {
    let lhs_stage_size = tiling_scheme.elements_in_stage_mk();
    let rhs_stage_size = tiling_scheme.elements_in_stage_nk();

    // u64 is the barrier, which is also in shared.
    // Just to ensure we don't go over by a few bytes accidentally.
    let inputs_stage_size_bytes = lhs_stage_size * size_of::<LhsS<MP>>() as u32
        + rhs_stage_size * size_of::<RhsS<MP>>() as u32
        + 2 * size_of::<u64>() as u32;
    let output_stage_size = tiling_scheme.elements_in_tile_mn() * num_planes;
    let output_stage_size_bytes = output_stage_size * size_of::<MP::EA>() as u32;

    let max_smem = client.properties().hardware.max_shared_memory_size;

    let max_stages = (max_smem as u32 - output_stage_size_bytes) / inputs_stage_size_bytes;
    let max_stages = Ord::min(max_stages, NUM_STAGES_MAX);

    let mut num_stages = prev_power_of_two(max_stages as u64) as u32;

    let num_tiles_k =
        (problem.k as u32).div_ceil(tiling_scheme.elements_in_stage_k()) / MIN_STAGES_PER_PIPELINE;

    while num_stages > num_tiles_k && num_stages > 1 {
        num_stages /= 2;
    }

    num_stages
}

/// Returns the greatest power of two less than or equal to `self`, or 0 otherwise.
pub const fn prev_power_of_two(n: u64) -> u64 {
    // n = 0 gives highest_bit_set_idx = 0.
    let highest_bit_set_idx = 63 - (n | 1).leading_zeros();
    // Binary AND of highest bit with n is a no-op, except zero gets wiped.
    (1 << highest_bit_set_idx) & n
}
