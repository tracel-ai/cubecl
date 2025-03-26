use crate::matmul::components::Ident;
use crate::matmul::components::global::IndexedQuantization;
use crate::matmul::components::global::multi_stage::AsyncBufferLoader;
use crate::matmul::components::global::multi_stage::BufferLoader;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::single_stage::AsyncBufferLoadingStrategy;
use crate::matmul::components::global::{self, CommonGlobalConfig};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
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
use crate::matmul::components::stage::single_buffer::{
    LhsBufferReaderFamily, RhsBufferReaderFamily,
};
use crate::matmul::kernels::MatmulAvailabilityError;

use super::AsyncLhsBufferLoader;
use super::AsyncRhsBufferLoader;

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
    SMM: stage::StageMatmulFamily<
            LhsReader = LhsBufferReaderFamily,
            RhsReader = RhsBufferReaderFamily,
        >,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingBarrierMatmul<
        MP,
        SMM::Matmul<MP::ES, MP::EG, MP::EA, LL::TilingLayout, RL::TilingLayout>,
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
    type Config = CommonGlobalConfig<SMM::Config>;

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

        CommonGlobalConfig::new(
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
    SMM: stage::StageMatmul<MP::ES, MP::EG, MP::EA>,
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
            MP::ES,
            MP::EG,
            MP::EA,
            LhsReader = LhsBufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = RhsBufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: AsyncBufferLoadingStrategy,
    RL: AsyncBufferLoadingStrategy,
{
    type Config = CommonGlobalConfig<SMM::Config>;
    type LhsLoader = AsyncLhsBufferLoader<MP::EG, MP::ES, SMM::Config, LL>;
    type RhsLoader = AsyncRhsBufferLoader<MP::EG, MP::ES, SMM::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        _quantization: CubeOption<IndexedQuantization<MP::EG>>,
        #[comptime] config: Self::Config,
    ) {
        let num_buffers = 2;
        let buffer_step = config.tiling_dimensions(Ident::Lhs).tile_shape_col();
        let k_step = num_buffers * buffer_step;

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        let lhs_buffer_reader_a = Self::LhsLoader::reader(&lhs_loader, BufferId::A);
        let rhs_buffer_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let lhs_buffer_reader_b = Self::LhsLoader::reader(&lhs_loader, BufferId::B);
        let rhs_buffer_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        let barrier_level = LL::barrier_level();
        comptime!(assert!(barrier_level == RL::barrier_level()));
        let barrier = Barrier::<MP::ES>::new(barrier_level);

        #[allow(clippy::collapsible_if)]
        if comptime!(config.check_k_bounds()) {
            if num_loops <= 1 {
                Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::A, config);
                Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::A, config);
                sync_units();
            }
        }

        Self::LhsLoader::fill_stage::<Barrier<MP::ES>>(
            &mut lhs_loader,
            &barrier,
            BufferId::A,
            config,
        );
        Self::RhsLoader::fill_stage::<Barrier<MP::ES>>(
            &mut rhs_loader,
            &barrier,
            BufferId::A,
            config,
        );

        for loop_iter in 0..num_loops {
            sync_units();

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::B, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::B, config);
                    sync_units();
                }
            }

            Self::LhsLoader::fill_stage::<Barrier<MP::ES>>(
                &mut lhs_loader,
                &barrier,
                BufferId::B,
                config,
            );
            Self::RhsLoader::fill_stage::<Barrier<MP::ES>>(
                &mut rhs_loader,
                &barrier,
                BufferId::B,
                config,
            );

            SMM::execute(
                &lhs_buffer_reader_a,
                &rhs_buffer_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                CubeOption::new_None(),
                config.to_smm_config(),
            );

            sync_units();
            barrier.wait();

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 2 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, BufferId::A, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, BufferId::A, config);
                    sync_units();
                }
            }

            Self::LhsLoader::fill_stage::<Barrier<MP::ES>>(
                &mut lhs_loader,
                &barrier,
                BufferId::A,
                config,
            );
            Self::RhsLoader::fill_stage::<Barrier<MP::ES>>(
                &mut rhs_loader,
                &barrier,
                BufferId::A,
                config,
            );

            SMM::execute(
                &lhs_buffer_reader_b,
                &rhs_buffer_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
                acc,
                CubeOption::new_None(),
                config.to_smm_config(),
            );
        }

        sync_units();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            CubeOption::new_None(),
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(rhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_unloader(
        out: VirtualTensor<MP::EG, ReadWrite>,
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
