use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg};
use cubecl_matmul::components::{
    AccG, AccS, InputRuntimeArg, LhsG, LhsS, MatmulSpec, OutputRuntimeArg, RhsG, RhsS,
    global::GlobalConfig as _,
    stage::{FullStageReaderFamily, StageMatmulFamily},
};
use cubecl_std::{FastDivmodArgs, tensor::layout::Coords2d};

use crate::{
    components::{
        ConvolutionProblem,
        global::{
            GlobalConfig,
            entry_point::{ConvolutionLaunch, implicit_conv, shape_divmod},
            single_stage::tma::SimpleTmaConvolutionFamily,
        },
    },
    kernels::layered::selector::RuntimeArgsLaunch,
};

impl<
    SMM: StageMatmulFamily<
            LhsStageReader = FullStageReaderFamily,
            RhsStageReader = FullStageReaderFamily,
            AccStageReader = Option<FullStageReaderFamily>,
            WriteCoords = Coords2d,
        >,
> ConvolutionLaunch<GlobalConfig<Self>> for SimpleTmaConvolutionFamily<SMM>
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
        let padded_channels =
            (problem.channels as u32).next_multiple_of(config.tiling_scheme().elements_in_tile_k());

        let size_k = problem.kernel_size.iter().product::<u32>() * padded_channels;

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(size_k),
            FastDivmodArgs::new(client, padded_channels),
            shape_divmod(client, &problem.out_shape),
            FastDivmodArgs::new(client, problem.channels as u32),
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
