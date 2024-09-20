use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::block_loop::matmul_execute;
use super::config::ComptimeCmmaInfo;
use super::runtime_info::{get_runtime_info, make_fragments, make_shared_memories};

#[cube(launch_unchecked)]
#[allow(unused_mut)]
pub fn cmma_kernel<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let runtime_info = get_runtime_info(lhs, rhs, out, comptime_info);
    let shared_memories = make_shared_memories::<FC>(comptime_info);
    let cmma_matrices = make_fragments::<F, FC>(comptime_info);

    matmul_execute::<F, FC>(
        lhs,
        rhs,
        out,
        shared_memories,
        cmma_matrices,
        runtime_info,
        comptime_info,
    );
}
