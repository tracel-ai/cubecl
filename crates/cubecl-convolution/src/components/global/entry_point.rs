use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{
    InputRuntimeArg, MatmulSpec, OutputRuntimeArg,
    global::{
        GlobalConfig as _,
        args::{MatmulArgs, TensorAcc, TensorLhs, TensorOutput, TensorRhs},
    },
};
use cubecl_std::{
    CubeOption, CubeOptionExpand, FastDivmod, FastDivmodArgs, tensor::r#virtual::VirtualTensor,
};

use crate::{
    components::{
        ConvolutionProblem,
        global::{GlobalConvolution, GlobalConvolutionFamily},
    },
    kernels::layered::selector::RuntimeArgs,
};

type Input<Args, Lhs, Rhs, EO> = <Args as MatmulArgs>::Input<Lhs, Rhs, EO>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

/// Provides launch entry point to solve a matmul
pub trait ConvolutionLaunch<Config> {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: &ConvolutionProblem,
        config: Config,
    );
}

#[cube(launch_unchecked)]
pub(crate) fn implicit_conv<
    Args: MatmulArgs,
    LhsG: Numeric,
    RhsG: Numeric,
    AccG: Numeric,
    LhsS: Numeric,
    RhsS: Numeric,
    AccS: Numeric,
    GMM: GlobalConvolutionFamily,
>(
    inputs: &Input<Args, LhsG, RhsG, AccG>,
    output: &mut Output<Args, AccG>,
    runtime_args: RuntimeArgs,
    #[comptime] config: GMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorLhs::<LhsG, RhsG, AccG, Args>::new(&state);
    let rhs = TensorRhs::<LhsG, RhsG, AccG, Args>::new(&state);
    let mut out = TensorOutput::<LhsG, RhsG, AccG, Args>::new(&mut state);

    let has_acc = Args::has_acc(&state);
    let bias: CubeOption<VirtualTensor<AccG>> = match has_acc {
        CubeOption::Some(_) => {
            let bias = TensorAcc::<LhsG, RhsG, AccG, Args>::new(&state);
            let bias = VirtualTensor::<AccG>::new::<TensorAcc<LhsG, RhsG, AccG, Args>>(&bias);
            CubeOption::new_Some(bias)
        }
        CubeOption::None => CubeOption::new_None(),
    };

    let lhs = VirtualTensor::<LhsG>::new::<TensorLhs<LhsG, RhsG, AccG, Args>>(&lhs);
    let rhs = VirtualTensor::<RhsG>::new::<TensorRhs<LhsG, RhsG, AccG, Args>>(&rhs);
    let out =
        VirtualTensor::<AccG, ReadWrite>::new::<TensorOutput<LhsG, RhsG, AccG, Args>>(&mut out);

    let stage_m = config.tiling_scheme().elements_in_stage_m().runtime();
    let stage_n = config.tiling_scheme().elements_in_stage_n().runtime();

    let m_offset = CUBE_POS_X * stage_m;
    let n_offset = CUBE_POS_Y * stage_n;

    let k_range = (0, runtime_args.shape_k);
    let k_size = runtime_args.shape_k;

    GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::execute(
        GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::init_lhs_loader(
            lhs,
            (0, m_offset, k_range.0),
            (1, stage_m, k_size),
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::init_rhs_loader(
            rhs,
            (0, k_range.0, n_offset),
            (1, k_size, stage_n),
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::init_bias_loader(
            bias, n_offset, stage_n, config,
        ),
        GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::init_writer(
            out,
            (0, m_offset, n_offset),
            (1, stage_m, stage_n),
            &runtime_args,
            config,
        ),
        &mut GMM::Convolution::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::init_accumulator(config),
        k_range,
        config,
    );
}

pub(crate) fn shape_divmod<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &[usize],
) -> SequenceArg<'a, R, FastDivmod> {
    let shape = shape
        .iter()
        .map(|s| FastDivmodArgs::new(client, *s as u32))
        .collect();
    SequenceArg { values: shape }
}
