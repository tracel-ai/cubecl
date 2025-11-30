use cubecl_core::{
    CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg, server::LaunchError,
};
use cubecl_matmul::components::{
    InputRuntimeArg, MatmulElems, OutputRuntimeArg,
    global::{PartitionedStageFamily, args::MatmulArgs},
    stage::{StageConfig as _, StageMatmulFamily, StridedStageFamily},
};
use cubecl_std::FastDivmodArgs;

use crate::components::{
    ConvolutionProblem,
    global::{
        GlobalConfig,
        args::RuntimeArgsLaunch,
        entry_point::{ConvolutionLaunch, implicit_conv, shape_divmod},
        single_stage::tma::SimpleTmaConvolutionFamily,
    },
};

impl<
    SMM: StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
> ConvolutionLaunch<GlobalConfig<Self>> for SimpleTmaConvolutionFamily<SMM>
{
    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        problem: &ConvolutionProblem,
        config: GlobalConfig<Self>,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError> {
        let padded_channels =
            (problem.channels as u32).next_multiple_of(config.stage_config.elements_in_tile_k());

        let size_k = problem.kernel_size.iter().product::<u32>() * padded_channels;

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(size_k),
            ScalarArg::new(problem.channels as u32),
            FastDivmodArgs::new(client, padded_channels),
            shape_divmod(client, &problem.out_shape),
        );

        unsafe {
            implicit_conv::launch_unchecked::<MA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                runtime_args,
                config,
                [*dtypes.lhs_global, *dtypes.rhs_global, *dtypes.acc_global],
                [*dtypes.lhs_stage, *dtypes.rhs_stage, *dtypes.acc_stage],
                [
                    *dtypes.lhs_register,
                    *dtypes.rhs_register,
                    *dtypes.acc_register,
                ],
            )
        }
    }
}
