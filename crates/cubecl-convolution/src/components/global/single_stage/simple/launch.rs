use cubecl_core::{
    CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg, server::LaunchError,
};
use cubecl_matmul::components::{
    InputRuntimeArg, MatmulElems, OutputRuntimeArg,
    global::{PartitionedStageFamily, args::MatmulArgs},
    stage::{StageMatmulFamily, StridedStageFamily},
};
use cubecl_std::FastDivmodArgs;

use crate::components::{
    ConvolutionProblem,
    global::{
        GlobalConfig,
        args::RuntimeArgsLaunch,
        entry_point::{ConvolutionLaunch, implicit_conv, shape_divmod},
        read::full_reader::FullLoadingStrategy,
        single_stage::simple::SimpleConvolutionFamily,
    },
};

impl<
    SMM: StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
    LL: FullLoadingStrategy,
    LR: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
> ConvolutionLaunch<GlobalConfig<Self>> for SimpleConvolutionFamily<SMM, LL, LR>
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
        let load_width = client.properties().hardware.load_width;
        let channel_align = load_width / dtypes.lhs_global.size_bits() as u32;
        let padded_channels = (problem.channels as u32).next_multiple_of(channel_align);

        let size_k = problem.kernel_size.iter().product::<u32>() * padded_channels;
        let padded_channels = FastDivmodArgs::new(client, padded_channels);

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(size_k),
            ScalarArg::new(problem.channels as u32),
            padded_channels,
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
