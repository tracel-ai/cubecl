use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::config::{ComptimeCmmaInfo, MainLoopStrategy};
use super::main_loop::{CmmaMain, SplitMainLoop, StandardMainLoop};
use crate::matmul::cmma::compute_loop::base::ComputeLoop;
use crate::matmul::cmma::compute_loop::{
    accumulators_first::AccumulatorsFirstComputeLoop,
    accumulators_first::AccumulatorsFirstWithReuseComputeLoop,
    buffers_first::BuffersFirstComputeLoop,
};
use crate::matmul::cmma::config::ComputeLoopOrderStrategy;

#[cube(launch_unchecked)]
pub fn cmma_launch<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    match comptime_info.main_loop_strategy {
        MainLoopStrategy::Standard => {
            cmma_build_step_1::<StandardMainLoop, F, FC>(lhs, rhs, out, comptime_info)
        }
        MainLoopStrategy::Split(_) => {
            cmma_build_step_1::<SplitMainLoop, F, FC>(lhs, rhs, out, comptime_info)
        }
    }
}

#[cube]
pub fn cmma_build_step_1<D: CmmaMain, F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    match comptime_info.compute_loop_order_strategy {
        ComputeLoopOrderStrategy::AllBuffersFirst => {
            cmma_execute::<BuffersFirstComputeLoop, D, F, FC>(lhs, rhs, out, comptime_info)
        }
        ComputeLoopOrderStrategy::AllAccumulatorsFirst(reuse_lhs_fragment) => {
            match reuse_lhs_fragment {
                false => cmma_execute::<AccumulatorsFirstComputeLoop, D, F, FC>(
                    lhs,
                    rhs,
                    out,
                    comptime_info,
                ),
                true => cmma_execute::<AccumulatorsFirstWithReuseComputeLoop, D, F, FC>(
                    lhs,
                    rhs,
                    out,
                    comptime_info,
                ),
            }
        }
    }
}

#[cube]
pub fn cmma_execute<C: ComputeLoop, D: CmmaMain, F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let (runtime_info, mut fragments, shared_memories) =
        D::prologue::<F, FC>(lhs, rhs, out, comptime_info);

    D::main_loop::<C, F, FC>(
        lhs,
        rhs,
        shared_memories,
        &mut fragments,
        runtime_info,
        comptime_info,
    );

    D::epilogue(out, fragments.accumulators, runtime_info, comptime_info);
}
