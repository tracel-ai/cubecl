use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::block_loop::{BlockLoop, SplitBlockLoop, StandardBlockLoop};
use super::config::ComptimeCmmaInfo;
use super::runtime_info::{get_runtime_info, make_fragments, make_shared_memories};

#[cube(launch_unchecked)]
pub fn cmma_launch<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    if comptime_info.role_id_strategy == 0 {
        cmma_kernel::<F, FC, StandardBlockLoop>(lhs, rhs, out, comptime_info);
    } else {
        cmma_kernel::<F, FC, SplitBlockLoop>(lhs, rhs, out, comptime_info);
    }
}

#[cube]
pub fn cmma_kernel<F: Float, FC: Float, D: BlockLoop>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let runtime_info = get_runtime_info::<F, D>(lhs, rhs, out, comptime_info);
    let shared_memories = make_shared_memories::<FC>(comptime_info);
    let cmma_matrices = make_fragments::<F, FC>(comptime_info);

    D::block_loop::<F, FC>(
        lhs,
        rhs,
        out,
        shared_memories,
        cmma_matrices,
        runtime_info,
        comptime_info,
    );
}
