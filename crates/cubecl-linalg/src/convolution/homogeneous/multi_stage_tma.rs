use std::{any::TypeId, marker::PhantomData};

use crate::{
    convolution::{
        ConvGemmConfig,
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
            MatmulSize, MatmulSpec, OutputRuntimeArg,
            global::{
                AccumulatorLoader, GlobalConfig, load::arrive_tma, output_loader::Unloader,
                single_stage,
            },
            stage::{FullReader, FullReaderFamily, StageConfig, StageMatmul, StageMatmulFamily},
        },
        kernels::MatmulAvailabilityError,
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
    implicit_conv,
};

/// Performs convolution at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
///
/// Uses multiple stages to prefetch as much data as can fit into shared memory, reducing the impact
/// of memory latency. An example execution would look like this:
///
/// * Start loading stage 1, 2, 3, 4
/// * Wait for stage 1
/// * Execute with stage 1 data
/// * Refill stage 1 with the data for stage 5
/// * Wait for stage 2
/// * Execute with stage 2 data
/// * Refill stage 2 with the data for stage 6
///
/// Keep going until k is exhausted
pub struct MultiStageTmaConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> Convolution<MP> for MultiStageTmaConvolution<MP, SMM>
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
        let num_stages = config.num_stages();
        let smm_config = config.to_smm_config();
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;
        // Loop once for each full set of stages, then once for each stage in an inner loop,
        // so the stage index is comptime. This is needed to make `Sequence` work.
        let num_loops = (num_loops + num_stages - 1) / num_stages;

        let total_stage_elems = config.tiling_dimensions(Ident::Rhs).total_size()
            + config.tiling_dimensions(Ident::Lhs).total_size();

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);

        sync_units();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(&mut acc_loader, acc, smm_config);

        let mut barriers = Sequence::<Barrier<MP::ES>>::new();
        let (mut tile_lhs, mut tile_rhs) = SMM::init_tile_inputs(smm_config);

        let mut stage = comptime![0u32];

        // Create barriers and prefetch each stage
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..num_stages {
            let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier, stage, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier, stage, smm_config);

            arrive_tma::<MP::ES>(&barrier, total_stage_elems);

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            barriers.push(barrier);

            comptime![stage += 1];
        }

        for k in 0..num_loops {
            let k = k * num_stages;

            let mut stage = comptime![0u32];

            // Loop through all stages
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..num_stages {
                let k = k + stage;
                let next_k = k + num_stages;

                // Bounds check for k stage, for when `k_stages % num_stages != 0`
                if k < k_range.1 {
                    let barrier = barriers.index(stage);

                    let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader, stage);
                    let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader, stage);

                    // Wait for load and execute matmul on this stage
                    barrier.wait();
                    SMM::execute(
                        lhs_stage_reader,
                        rhs_stage_reader,
                        &mut tile_lhs,
                        &mut tile_rhs,
                        acc,
                        config.to_smm_config(),
                    );
                    barrier.arrive();

                    // Check if there's any stages left to load in the k dimension
                    if next_k < k_range.1 {
                        barrier.wait();

                        // Refill stage and advance view
                        Self::LhsLoader::fill_stage(&mut lhs_loader, barrier, stage, config);
                        Self::RhsLoader::fill_stage(&mut rhs_loader, barrier, stage, smm_config);

                        arrive_tma::<MP::ES>(barrier, total_stage_elems);

                        Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
                        Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
                    }
                }

                comptime![stage += 1];
            }
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
        Self::LhsLoader::new(
            lhs,
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(),
            config,
        )
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
            config.num_stages(),
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

pub struct MultiStageTmaConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

impl<SMM> ConvolutionFamily for MultiStageTmaConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
{
    type Convolution<MP: MatmulPrecision> =
        MultiStageTmaConvolution<MP, SMM::Matmul<MP, TmaIm2colTiling, TmaWeightTiling>>;
}

impl<SMM> ConvolutionConfigFactory for MultiStageTmaConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily,
{
    type Config = config::ConvolutionConfig<single_stage::Config<SMM::Config>>;
    type Input = SMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
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
            input,
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            false,
        );
        let size = SMM::stage_shape(&smm_config);
        let shape = SMM::tile_shape(&smm_config);

        let num_stages =
            num_stages::<R, MP>(client, &size, &shape, &problem, smm_config.num_planes());

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
            ),
            problem.kernel_size,
            problem.stride,
            problem.dilation,
            problem.padding,
            num_stages,
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
    ConvolutionLaunch for MultiStageTmaConvolutionFamily<SMM>
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

        let size_m = problem.batches * problem.out_h * problem.out_w;
        let size_n = problem.n;
        let size_k = config.kernel_size(0) * config.kernel_size(1) * padded_channels;

        let runtime_args = RuntimeArgsLaunch::new(
            ScalarArg::new(size_m as u32),
            ScalarArg::new(size_n as u32),
            ScalarArg::new(size_k),
            FastDivmodArgs::new(client, padded_channels),
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

/// More than 4 stages would likely slow things down from code size
/// Should test more to find the ideal value here, just using 4 because that's what cuDNN uses
const NUM_STAGES_MAX: u32 = 8;
/// I found that too many pipeline stages relative to k degrade performance
const MIN_STAGES_PER_PIPELINE: u32 = 32;

fn num_stages<R: Runtime, MP: MatmulPrecision>(
    client: &ComputeClient<R::Server, R::Channel>,
    stage_size: &MatmulSize,
    tile_size: &MatmulSize,
    problem: &ConvolutionProblem,
    num_planes: u32,
) -> u32 {
    let inputs_stage_size = stage_size.m * stage_size.k + stage_size.k * stage_size.n;
    // u64 is the barrier, which is also in shared.
    // Just to ensure we don't go over by a few bytes accidentally.
    let inputs_stage_size_bytes =
        inputs_stage_size * size_of::<MP::ES>() as u32 + size_of::<u64>() as u32;
    let output_stage_size = tile_size.m * tile_size.n * num_planes;
    let output_stage_size_bytes = output_stage_size * size_of::<MP::EA>() as u32;

    let max_smem = client
        .properties()
        .hardware_properties()
        .max_shared_memory_size;

    let max_stages = (max_smem as u32 - output_stage_size_bytes) / inputs_stage_size_bytes;
    let max_stages = Ord::min(max_stages, NUM_STAGES_MAX);

    let mut num_stages = prev_power_of_two(max_stages as u64) as u32;

    let num_tiles_k = (problem.k as u32).div_ceil(stage_size.k) / MIN_STAGES_PER_PIPELINE;

    while num_stages > num_tiles_k && num_stages > 1 {
        num_stages /= 2;
    }

    // println!(
    //     "max_stages: {max_stages}, num_stages: {num_stages}, max_smem: {max_smem}, stage_in: {inputs_stage_size_bytes}, stage_out: {output_stage_size_bytes}"
    // );

    num_stages
}

/// Returns the greatest power of two less than or equal to `self`, or 0 otherwise.
pub const fn prev_power_of_two(n: u64) -> u64 {
    // n = 0 gives highest_bit_set_idx = 0.
    let highest_bit_set_idx = 63 - (n | 1).leading_zeros();
    // Binary AND of highest bit with n is a no-op, except zero gets wiped.
    (1 << highest_bit_set_idx) & n
}
