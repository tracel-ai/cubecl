use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg};
use cubecl_matmul::components::{
    EA, EO, InputRuntimeArg, LhsG, LhsS, MatmulSpec, OutputRuntimeArg, RhsG, RhsS,
    stage::{FullReaderFamily, StageMatmulFamily},
};
use cubecl_std::{FastDivmodArgs, tensor::layout::Coords3d};

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
            LhsReader = FullReaderFamily,
            RhsReader = FullReaderFamily,
            WriteCoords = Coords3d,
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
                LhsS<MS>,
                RhsS<MS>,
                EA<MS>,
                EO<MS>,
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
