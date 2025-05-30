use std::marker::PhantomData;

use crate::{
    convolution::{
        base::{
            Convolution, ConvolutionConfigFactory, ConvolutionFamily, ConvolutionLaunch,
            ConvolutionProblem, RuntimeArgs, RuntimeArgsLaunch,
        },
        loader::{bias::BiasLoader, im2col::SimpleIm2colLoader},
    },
    matmul::{
        components::{
            EA, EI, EO, ES, InputIdent, InputRuntimeArg, InvalidConfigError, MatmulLineSizes,
            MatmulPrecision, MatmulSpec, OutputRuntimeArg,
            global::{
                AccumulatorLoader, GlobalConfig,
                load::{SyncFullLoader, sync_full_cyclic},
                single_stage,
            },
            stage::{
                ContiguousTilingLayout, FullReaderFamily, FullStageToTileReader,
                RowMajorTilingOrder, StageConfig, StageMatmul, StageMatmulFamily,
            },
        },
        kernels::matmul::GlobalInput,
    },
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, FastDivmodArgs,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::base::{
    config::{self, ConvolutionConfig},
    implicit_conv, shape_divmod,
};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimpleConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> Convolution<MP> for SimpleConvolution<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsReader = FullStageToTileReader<MP::ES, ConvTilingLayout>,
            RhsReader = FullStageToTileReader<MP::ES, ConvTilingLayout>,
        >,
{
    type LhsLoader = SimpleIm2colLoader<MP, Self::Config>;
    type Config = ConvolutionConfig<single_stage::Config<SMM::Config>>;
    type RhsLoader =
        SyncFullLoader<MP, Self::Config, sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>>;
    type AccumulatorLoader = BiasLoader<MP>;

    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());

        sync_cube();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.stage_config(),
        );

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            sync_cube();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        sync_cube();

        SMM::write_results::<Self::Config>(acc, &mut out_writer, config.stage_config(), config);
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(lhs, x_offset, y_offset, runtime_args, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(
            rhs,
            x_offset,
            y_offset,
            0,
            CubeOption::new_None(),
            InputIdent::Rhs,
            config,
        )
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<MP::EO>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccumulatorLoader {
        Self::AccumulatorLoader::new::<Self::Config>(bias, n_offset, config)
    }

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Writer {
        SMM::init_writer(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}

pub struct SimpleConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

impl<SMM> ConvolutionFamily for SimpleConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
{
    type Convolution<MP: MatmulPrecision> =
        SimpleConvolution<MP, SMM::Matmul<MP, ConvTilingLayout, ConvTilingLayout>>;
}

impl<SMM> ConvolutionConfigFactory for SimpleConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily,
{
    type Config = config::ConvolutionConfig<single_stage::Config<SMM::Config>>;
    type Input = GlobalInput<SMM::Input>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.stage_config())
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        _client: &ComputeClient<R::Server, R::Channel>,
        input: Self::Input,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Self::Config {
        let stage_config = SMM::make_config(
            input.stage_input,
            &problem.as_matmul_problem(),
            line_sizes,
            cube_dim,
            cube_count,
            false,
        );
        let stage_k = stage_config.tiling_scheme().elements_in_stage_k();

        config::ConvolutionConfig::new(
            single_stage::Config::new(
                stage_config,
                // TODO: Find the correct condition to avoid check bounds.
                true,
                true,
                true,
                problem.lhs_layout,
                problem.rhs_layout,
                line_sizes.lhs as u32,
                line_sizes.rhs as u32,
                line_sizes.out as u32,
                stage_k,
                input.loading_precompute_strategy,
                input.loader_mode,
            ),
            &problem.kernel_size,
            &problem.stride,
            &problem.dilation,
            &problem.padding,
            problem.dimensionality,
            1,
        )
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), crate::matmul::kernels::MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config())
    }
}

impl<SMM: StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>>
    ConvolutionLaunch for SimpleConvolutionFamily<SMM>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        bias: Option<TensorArg<'a, R>>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: &ConvolutionProblem,
        config: <Self as ConvolutionConfigFactory>::Config,
    ) {
        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(problem.k as u32),
            FastDivmodArgs::new(client, problem.channels as u32),
            shape_divmod(client, &problem.out_shape),
        );

        unsafe {
            implicit_conv::launch_unchecked::<MS::Args, EI<MS>, ES<MS>, EA<MS>, EO<MS>, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                bias.into(),
                output,
                runtime_args,
                config,
            );
        }
    }
}
