use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg};
use cubecl_matmul::components::{
    AccG, AccS, InputRuntimeArg, LhsG, LhsS, MatmulSpec, OutputRuntimeArg, RhsG, RhsS,
    global::PartitionedStageFamily,
    stage::{StageMatmulFamily, StridedStageFamily},
};
use cubecl_std::FastDivmodArgs;

use crate::{
    components::{
        ConvolutionProblem,
        global::{
            GlobalConfig,
            entry_point::{ConvolutionLaunch, implicit_conv, shape_divmod},
            single_stage::simple::SimpleConvolutionFamily,
        },
    },
    kernels::layered::selector::RuntimeArgsLaunch,
};

impl<
    SMM: StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
> ConvolutionLaunch<GlobalConfig<Self>> for SimpleConvolutionFamily<SMM>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: &ConvolutionProblem,
        config: GlobalConfig<Self>,
    ) {
        let shape_channels = FastDivmodArgs::new(client, problem.channels as u32);

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(problem.k as u32),
            shape_channels,
            shape_divmod(client, &problem.out_shape),
            shape_channels,
        );

        unsafe {
            implicit_conv::launch_unchecked::<
                MS::Args,
                LhsG<MS>,
                RhsG<MS>,
                AccG<MS>,
                LhsS<MS>,
                RhsS<MS>,
                AccS<MS>,
                Self,
                R,
            >(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                runtime_args,
                config,
            );
        }
    }
}
