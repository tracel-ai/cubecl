use std::{any::TypeId, marker::PhantomData};

use crate::{
    convolution::{
        base::{
            Convolution, ConvolutionConfigFactory, ConvolutionFamily, ConvolutionLaunch,
            ConvolutionProblem, RuntimeArgs, RuntimeArgsLaunch,
        },
        loader::{
            bias::BiasLoader,
            im2col_tma::{TmaIm2colLoader, TmaIm2colTiling},
            weight_tma::{TmaWeightLoader, TmaWeightTiling},
        },
    },
    matmul::{
        components::{
            EA, EI, EO, ES, Ident, InputRuntimeArg, InvalidConfigError, MatmulPrecision,
            MatmulSpec, OutputRuntimeArg,
            global::{
                AccumulatorLoader, GlobalConfig, TilewiseWriter, load::arrive_tma, single_stage,
            },
            stage::{FullReader, FullReaderFamily, StageMatmul, StageMatmulFamily},
        },
        kernels::{MatmulAvailabilityError, matmul::LoadingPrecomputeStrategy},
    },
};
use cubecl_core::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
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
pub struct SimpleTmaConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> Convolution<MP> for SimpleTmaConvolution<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsReader = FullReader<MP::ES, TmaIm2colTiling>,
            RhsReader = FullReader<MP::ES, TmaWeightTiling>,
        >,
{
    type LhsLoader = TmaIm2colLoader<MP, Self::Config>;
    type Config = ConvolutionConfig<single_stage::Config<SMM::Config>>;
    type RhsLoader = TmaWeightLoader<MP, SMM::Config>;
    type AccumulatorLoader = BiasLoader<MP>;

    type Out = TilewiseWriter<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_writer: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        let total_stage_elems = config.tiling_dimensions(Ident::Rhs).total_size()
            + config.tiling_dimensions(Ident::Lhs).total_size();

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        sync_cube();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.to_smm_config(),
        );

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier, 0u32, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier, 0u32, config.to_smm_config());

            arrive_tma::<MP::ES>(&barrier, total_stage_elems);

            barrier.wait();

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader, 0u32);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader, 0u32);

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

        sync_cube();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_writer,
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
        Self::LhsLoader::new(lhs, x_offset, y_offset, runtime_args, 1u32, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            CubeOption::new_None(),
            runtime_args,
            1u32,
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
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }
}

pub struct SimpleTmaConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

impl<SMM> ConvolutionFamily for SimpleTmaConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
{
    type Convolution<MP: MatmulPrecision> =
        SimpleTmaConvolution<MP, SMM::Matmul<MP, TmaIm2colTiling, TmaWeightTiling>>;
}

impl<SMM> ConvolutionConfigFactory for SimpleTmaConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily,
{
    type Config = config::ConvolutionConfig<single_stage::Config<SMM::Config>>;
    type Input = (SMM::Input, LoadingPrecomputeStrategy);

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        _client: &ComputeClient<R::Server, R::Channel>,
        input: Self::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Self::Config {
        let mut problem = problem.clone();

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        problem.lhs_line_size = 1;
        problem.rhs_line_size = 1;

        let smm_config = SMM::make_config(
            input.0,
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            false,
        );
        let size = SMM::stage_shape(&smm_config);

        config::ConvolutionConfig::new(
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
                input.1,
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
    ) -> Result<(), MatmulAvailabilityError> {
        let id_ei = TypeId::of::<MP::EI>();
        let id_es = TypeId::of::<MP::ES>();
        let is_tf32 = id_ei == TypeId::of::<f32>() && id_es == TypeId::of::<tf32>();

        if id_ei != id_es && !is_tf32 {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
    }
}

impl<SMM: StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>>
    ConvolutionLaunch for SimpleTmaConvolutionFamily<SMM>
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
        let tiling_dims = config.tiling_dimensions(Ident::Lhs);
        let padded_channels =
            (problem.channels as u32).next_multiple_of(tiling_dims.tile_shape_col());

        let size_k = problem.kernel_size.iter().product::<u32>() * padded_channels;

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(problem.m as u32),
            ScalarArg::new(problem.n as u32),
            ScalarArg::new(size_k),
            FastDivmodArgs::new(client, padded_channels),
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
