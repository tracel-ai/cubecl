use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::MatmulElems;
use cubecl_matmul::components::global::GlobalConfig;
use cubecl_matmul::components::stage::StageConfig as _;
use cubecl_matmul::components::{
    InputRuntimeArg, OutputRuntimeArg, batch::SliceIndex, global::args::MatmulArgs,
};
use cubecl_std::{CubeOption, CubeOptionExpand, FastDivmod, FastDivmodArgs};

use crate::components::{ConvGemmConfig, global::args::RuntimeArgs};
use crate::components::{
    ConvolutionProblem,
    global::{GlobalConvolution, GlobalConvolutionFamily},
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
    unsafe fn launch_unchecked<'a, MA: MatmulArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MA, R>,
        output: OutputRuntimeArg<'a, MA, R>,
        problem: &ConvolutionProblem,
        config: Config,
        dtypes: &MatmulElems,
    ) -> Result<(), LaunchError>;
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
    LhsR: Numeric,
    RhsR: Numeric,
    AccR: Numeric,
    GMM: GlobalConvolutionFamily,
>(
    inputs: &Input<Args, LhsG, RhsG, AccG>,
    output: &mut Output<Args, AccG>,
    runtime_args: RuntimeArgs,
    #[comptime] config: GMM::Config,
    #[define(LhsG, RhsG, AccG)] _global: [StorageType; 3],
    #[define(LhsS, RhsS, AccS)] _stage: [StorageType; 3],
    #[define(LhsR, RhsR, AccR)] _register: [StorageType; 3],
) {
    let mut state = Args::init_state::<
        LhsG,
        RhsG,
        AccG,
        <GMM::Config as ConvGemmConfig>::GlobalMatmulConfig,
    >(inputs, output, config.matmul_config());

    let lhs = Args::view_lhs(&state);
    let rhs = Args::view_rhs(&state);
    let bias = Args::view_acc(&state);
    let out = Args::view_out(&mut state);

    let stage_m = config
        .matmul_config()
        .stage_config()
        .elements_in_stage_m()
        .runtime();
    let stage_n = config
        .matmul_config()
        .stage_config()
        .elements_in_stage_n()
        .runtime();

    let m_offset = CUBE_POS_X * stage_m;
    let n_offset = CUBE_POS_Y * stage_n;

    let k_range = (0, runtime_args.shape_k);
    let k_size = runtime_args.shape_k;

    let lhs = lhs.view(SliceIndex::new(0, lhs.shape()));
    let rhs = rhs.view(SliceIndex::new(0, rhs.shape()));
    let bias = match bias {
        CubeOption::Some(bias) => {
            let view = bias.view(SliceIndex::new(0, bias.shape()));
            CubeOption::new_Some(view.slice_unchecked((0, n_offset), (1, stage_n)))
        }
        CubeOption::None => CubeOption::new_None(),
    };
    let out = out.view_mut(SliceIndex::new(0, out.shape()));

    GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::execute(
        GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::init_lhs_global_reader(
            lhs,
            (m_offset, k_range.0),
            (stage_m, k_size),
            &runtime_args,
            config,
        ),
        GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::init_rhs_global_reader(
            rhs.slice_unchecked((k_range.0, n_offset), (k_size, stage_n)),
            &runtime_args,
            config,
        ),
        GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::init_bias_global_reader(
            bias, config,
        ),
        GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::init_global_writer(
            out.slice_mut_unchecked((m_offset, n_offset), (stage_m, stage_n)),
            config,
        ),
        &mut GMM::Convolution::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::init_accumulator(
            config,
        ),
        k_range,
        config,
    );
}

pub(crate) fn shape_divmod<'a, R: Runtime>(
    client: &ComputeClient<R>,
    shape: &[usize],
) -> SequenceArg<'a, R, FastDivmod> {
    shape
        .iter()
        .map(|s| FastDivmodArgs::new(client, *s as u32))
        .collect()
}
