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
    CubeOption, CubeOptionExpand, FastDivmod, FastDivmodArgs,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
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
    LhsS: Numeric,
    RhsS: Numeric,
    EA: Numeric,
    EO: Numeric,
    GMM: GlobalConvolutionFamily,
>(
    inputs: &Input<Args, LhsG, RhsG, EO>,
    output: &mut Output<Args, EO>,
    runtime_args: RuntimeArgs,
    #[comptime] config: GMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorLhs::<LhsG, RhsG, EO, Args>::new(&state);
    let rhs = TensorRhs::<LhsG, RhsG, EO, Args>::new(&state);
    let mut out = TensorOutput::<LhsG, RhsG, EO, Args>::new(&mut state);

    let has_acc = Args::has_acc(&state);
    let bias: CubeOption<TensorAcc<LhsG, RhsG, EO, Args>> = match has_acc {
        CubeOption::Some(_) => CubeOption::new_Some(TensorAcc::<LhsG, RhsG, EO, Args>::new(&state)),
        CubeOption::None => CubeOption::new_None(),
    };

    let lhs = VirtualTensor::<LhsG>::new::<TensorLhs<LhsG, RhsG, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<RhsG>::new::<TensorRhs<LhsG, RhsG, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<LhsG, RhsG, EO, Args>>(&mut out);

    let bias = match &bias {
        CubeOption::Some(bias) => CubeOption::new_Some(VirtualTensor::<EO>::new::<
            TensorAcc<LhsG, RhsG, EO, Args>,
        >(bias)),
        CubeOption::None => CubeOption::new_None(),
    };

    let x_offset = CUBE_POS_X * config.tiling_scheme().elements_in_stage_m();
    let y_offset = CUBE_POS_Y * config.tiling_scheme().elements_in_stage_n();
    let k_range = (0, runtime_args.size_k);

    GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::execute(
        GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::init_lhs_loader(
            lhs,
            x_offset,
            k_range.0,
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::init_rhs_loader(
            rhs,
            k_range.0,
            y_offset,
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::init_bias_loader(
            bias, y_offset, config,
        ),
        GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::init_writer(
            out,
            x_offset,
            y_offset,
            &runtime_args,
            config,
        ),
        &mut GMM::Convolution::<(LhsG, RhsG, LhsS, RhsS, EA, EO)>::init_accumulator(config),
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
