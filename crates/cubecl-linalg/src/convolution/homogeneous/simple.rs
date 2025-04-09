use std::marker::PhantomData;

use crate::{
    convolution::{
        ConvGemmConfig,
        base::{
            Convolution, ConvolutionConfigFactory, ConvolutionFamily, ConvolutionLaunch,
            ConvolutionProblem, RuntimeArgs, RuntimeArgsLaunch,
        },
        loader::{bias::BiasLoader, im2col::SimpleIm2colLoader},
    },
    matmul::components::{
        EA, EI, EO, ES, InputIdent, InputRuntimeArg, InvalidConfigError, MatmulPrecision,
        MatmulSpec, OutputRuntimeArg,
        global::{
            AccumulatorLoader, GlobalConfig,
            load::{SyncFullLoader, sync_full_cyclic},
            output_loader::Unloader,
            single_stage,
        },
        stage::{
            ContiguousTilingLayout, RowMajorTilingOrder, StageMatmul, StageMatmulFamily,
            multi_buffer::{FullReader, FullReaderFamily},
        },
    },
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, FastDivmodArgs,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::base::{
    config::{self, HomogeneousConfig},
    implicit_conv,
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
            LhsReader = FullReader<MP::ES, ConvTilingLayout>,
            RhsReader = FullReader<MP::ES, ConvTilingLayout>,
        >,
{
    type LhsLoader = SimpleIm2colLoader<MP, Self::Config>;
    type Config = HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type RhsLoader =
        SyncFullLoader<MP, SMM::Config, sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>>;
    type AccumulatorLoader = BiasLoader<MP>;

    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        Self::AccumulatorLoader::fill_stage::<SMM::Config>(&mut acc_loader, config.to_smm_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        sync_units();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.to_smm_config(),
        );

        for _ in 0..num_loops {
            sync_units();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config.to_matmul_config());

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        sync_units();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
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
        Self::RhsLoader::new::<Self::Config>(
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
        Self::AccumulatorLoader::new::<SMM::Config>(bias, n_offset, config.to_smm_config())
    }

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
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
    type Config = config::HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type Input = SMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn make_config(
        input: Self::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Self::Config {
        let smm_config = SMM::make_config(
            input,
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            false,
        );
        let size = SMM::stage_shape(&smm_config);

        config::HomogeneousConfig::new(
            single_stage::Config::new(
                smm_config,
                // TODO: Find the correct condition to avoid check bounds.
                true,
                true,
                true,
                problem.lhs_layout,
                problem.rhs_layout,
                problem.lhs_line_size as u32,
                problem.rhs_line_size as u32,
                problem.out_line_size as u32,
                size.k,
            ),
            problem.kernel_size,
            problem.stride,
            problem.dilation,
            problem.padding,
        )
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), crate::matmul::kernels::MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
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
        let size_m = problem.batches * problem.out_h * problem.out_w;
        let size_n = problem.n;
        let size_k = config.kernel_size(0) * config.kernel_size(1) * problem.channels as u32;

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(size_m as u32),
            ScalarArg::new(size_n as u32),
            ScalarArg::new(size_k),
            FastDivmodArgs::new(client, problem.channels as u32),
            FastDivmodArgs::new(client, problem.out_h as u32),
            FastDivmodArgs::new(client, problem.out_w as u32),
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
