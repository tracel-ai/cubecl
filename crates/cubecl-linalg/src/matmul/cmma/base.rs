use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::config::ComptimeCmmaInfo;
use super::main_loop::{MainLoop, SplitMainLoop, StandardMainLoop};
use super::prologue::{get_runtime_info, make_fragments, make_shared_memories};

#[cube(launch_unchecked)]
pub fn cmma_launch<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    if comptime_info.main_loop_strategy == 0 {
        cmma_kernel::<F, FC, StandardMainLoop>(lhs, rhs, out, comptime_info);
    } else {
        cmma_kernel::<F, FC, SplitMainLoop>(lhs, rhs, out, comptime_info);
    }
}

#[cube]
pub fn cmma_kernel<F: Float, FC: Float, D: MainLoop>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let runtime_info = get_runtime_info::<F, D>(lhs, rhs, out, comptime_info);
    let shared_memories = make_shared_memories::<FC>(comptime_info);
    let mut fragments = make_fragments::<F, FC>(comptime_info);

    D::main_loop::<F, FC>(
        lhs,
        rhs,
        shared_memories,
        &mut fragments,
        runtime_info,
        comptime_info,
    );

    D::epilogue(out, fragments.accumulators, runtime_info, comptime_info);
}
