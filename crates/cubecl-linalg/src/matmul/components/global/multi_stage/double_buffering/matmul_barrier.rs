use crate::matmul::components::Ident;
use crate::matmul::components::InputIdent;
use crate::matmul::components::global;
use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::{
    AsyncBufferLoader, AsyncBufferLoadingStrategy, BufferId,
};
use crate::matmul::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::BufferReader;
use crate::matmul::components::stage::StageMemory;
use crate::matmul::components::{MatmulPrecision, stage};
use cubecl_core::Feature;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use std::marker::PhantomData;

use crate::matmul::components::InvalidConfigError;
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::global::GlobalMatmulFamily;
use crate::matmul::components::stage::BufferReaderFamily;
use crate::matmul::kernels::MatmulAvailabilityError;

pub struct DoubleBufferingBarrierMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = BufferReaderFamily, RhsReader = BufferReaderFamily>,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingBarrierMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>,
        LL,
        RL,
    >;
}

impl<SMM, LL, RL> MatmulConfigFactory for DoubleBufferingBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
{
    type Input = SMM::Input;
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;

        if config.tiling_dimensions(Ident::Lhs).tile_count_col() != 2 {
            return Err(Box::new("Double buffering matmul needs exactly 2 buffers."));
        }

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        if !client.properties().feature_enabled(Feature::Barrier) {
            return Err(MatmulAvailabilityError::BarrierUnavailable);
        }

        SMM::check_availability::<R, MP>(client, &config.smm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let smm_config = SMM::make_config(input, problem, cube_dim, cube_count, quantized);
        let stage_shape = SMM::stage_shape(&smm_config);

        DoubleBufferingGlobalConfig::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % stage_shape.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
        )
    }
}

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct DoubleBufferingBarrierMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> global::GlobalMatmul<MP>
    for DoubleBufferingBarrierMatmul<MP, SMM, LL, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = BufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = BufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = AsyncBufferLoader<MP, SMM::Config, Barrier<MP::ES>, LL>;
    type RhsLoader = AsyncBufferLoader<MP, SMM::Config, Barrier<MP::ES>, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let num_stages = 2;
        let buffer_step = config.tiling_dimensions(Ident::Lhs).tile_shape_col();
        let k_step = num_stages * buffer_step;

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages - 1;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        let lhs_buffer_reader_a = Self::LhsLoader::reader(&lhs_loader, BufferId::A);
        let rhs_buffer_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let lhs_buffer_reader_b = Self::LhsLoader::reader(&lhs_loader, BufferId::B);
        let rhs_buffer_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        let barrier_level = LL::barrier_level();
        let barrier_a = Barrier::<MP::ES>::new(barrier_level);
        let barrier_b = Barrier::<MP::ES>::new(barrier_level);

        #[allow(clippy::collapsible_if)]
        if comptime!(config.check_k_bounds()) {
            if num_loops == 0 {
                Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::A, config);
                Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::A, config);
                sync_units();
            }
        }
        Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier_a, BufferId::A, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier_a, BufferId::A, config);
        barrier_a.arrive();

        // So it can do the first iteration
        barrier_b.arrive();

        for loop_iter in 0..num_loops {
            barrier_b.wait();
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier_b, BufferId::B, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier_b, BufferId::B, config);
            barrier_b.arrive();

            barrier_a.wait();
            SMM::execute(
                &lhs_buffer_reader_a,
                &rhs_buffer_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                config.to_smm_config(),
            );
            barrier_a.arrive();

            barrier_b.wait();
            SMM::execute(
                &lhs_buffer_reader_b,
                &rhs_buffer_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
                acc,
                config.to_smm_config(),
            );
            barrier_b.arrive();

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            barrier_a.wait();
            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::A, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::A, config);
                    // TODO can we remove
                    sync_units();
                }
            }
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier_a, BufferId::A, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier_a, BufferId::A, config);
            barrier_a.arrive();
        }

        barrier_b.wait();
        #[allow(clippy::collapsible_if)]
        if comptime!(config.check_k_bounds()) {
            Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::B, config);
            Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::B, config);
            // TODO can we remove
            sync_units();
        }
        Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier_b, BufferId::B, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier_b, BufferId::B, config);
        barrier_b.arrive();

        barrier_a.wait();
        SMM::execute(
            &lhs_buffer_reader_a,
            &rhs_buffer_reader_a,
            &mut lhs_tile_a,
            &mut rhs_tile_a,
            acc,
            config.to_smm_config(),
        );
        barrier_a.arrive();

        barrier_b.wait();
        SMM::execute(
            &lhs_buffer_reader_b,
            &rhs_buffer_reader_b,
            &mut lhs_tile_b,
            &mut rhs_tile_b,
            acc,
            config.to_smm_config(),
        );
        barrier_b.arrive();

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
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        let tensor_reader = TensorReader::new(lhs, x_offset, y_offset, batch_offset);
        let stage = StageMemory::new::<SMM::Config>(2u32, Ident::Lhs, config.to_smm_config());

        Self::LhsLoader::new(tensor_reader, stage, quantization, InputIdent::Lhs, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        let tensor_reader = TensorReader::new(rhs, x_offset, y_offset, batch_offset);
        let stage = StageMemory::new::<SMM::Config>(2u32, Ident::Rhs, config.to_smm_config());

        Self::RhsLoader::new(tensor_reader, stage, quantization, InputIdent::Rhs, config)
    }

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        SMM::zero_accumulator(acc, config.to_smm_config());
    }
}
