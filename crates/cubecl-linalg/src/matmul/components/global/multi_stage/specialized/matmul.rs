use crate::matmul::components::{
    Ident, MatmulPrecision,
    global::{
        self, GlobalMatmul, IndexedQuantization, ZeroAccumulatorLoader,
        multi_stage::{
            BufferLoader, SyncBufferLoader, SyncBufferLoadingStrategy, double_buffering::BufferId,
        },
        output_loader::Unloader,
    },
    stage::{
        StageMatmul,
        single_buffer::{LhsBufferReader, RhsBufferReader},
    },
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::config::Config;
use super::loader::{SyncLhsBufferLoader, SyncRhsBufferLoader};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};

use crate::matmul::{
    components::{
        InvalidConfigError, MatmulConfigFactory, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{
            self,
            single_buffer::{LhsBufferReaderFamily, RhsBufferReaderFamily},
        },
    },
    kernels::MatmulAvailabilityError,
};

pub struct SpecializedMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SpecializedMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<
            LhsReader = LhsBufferReaderFamily,
            RhsReader = RhsBufferReaderFamily,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = SpecializedMatmul<
        MP,
        SMM::Matmul<MP::ES, MP::EG, MP::EA, LL::TilingLayout, RL::TilingLayout>,
        LL,
        RL,
    >;
}

impl<SMM, LL, RL> MatmulConfigFactory for SpecializedMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Input = SMM::Input;
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        if config.num_producers() == 0 {
            return Err(Box::new(
                "There are no producer planes. Make sure there are more planes than the underlying stage matmul requires.",
            ));
        }
        if config.tiling_dimensions(Ident::Lhs).tile_count_col() <= 1 {
            return Err(Box::new("Producer-consumer needs at least 2 buffers."));
        }

        LL::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
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

        Config::new(
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

/// Performs matrix multiplication at the global level, with planes split between two roles:
/// - First n planes are used in the stage matmul computation, with n the number of planes needed by the underlying stage matmul
/// - Remaining planes load data to the stage
///
/// Both roles alternate the buffer (tile index in dimension k) they are working on
pub struct SpecializedMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP::ES, MP::EG, MP::EA>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> global::GlobalMatmul<MP>
    for SpecializedMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
            MP::ES,
            MP::EG,
            MP::EA,
            LhsReader = LhsBufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = RhsBufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = SyncLhsBufferLoader<MP::EG, MP::ES, SMM::Config, LL>;
    type RhsLoader = SyncRhsBufferLoader<MP::EG, MP::ES, SMM::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<IndexedQuantization<MP::EG>>,
        #[comptime] config: Self::Config,
    ) {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let is_consumer = Self::is_consumer(config);

        let num_buffers = config.tiling_dimensions(Ident::Lhs).tile_count_col();
        let buffer_step = config.tiling_dimensions(Ident::Lhs).tile_shape_col();
        let k_step = num_buffers * buffer_step;

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages * num_buffers;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        let lhs_buffer_reader_a = Self::LhsLoader::reader(&lhs_loader, BufferId::A);
        let rhs_buffer_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let lhs_buffer_reader_b = Self::LhsLoader::reader(&lhs_loader, BufferId::B);
        let rhs_buffer_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        for _ in 0..num_loops {
            Self::LhsLoader::fill_stage(&mut lhs_loader, BufferId::A, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, BufferId::A, config);

            sync_units();

            if is_consumer {
                SMM::execute(
                    &lhs_buffer_reader_a,
                    &rhs_buffer_reader_a,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    acc,
                    CubeOption::new_None(),
                    config.to_smm_config(),
                );
            }

            Self::LhsLoader::fill_stage(&mut lhs_loader, BufferId::B, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, BufferId::B, config);

            sync_units();

            if is_consumer {
                SMM::execute(
                    &lhs_buffer_reader_b,
                    &rhs_buffer_reader_b,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    acc,
                    CubeOption::new_None(),
                    config.to_smm_config(),
                );
            }

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        if is_consumer {
            SMM::read_accumulator::<Self::Out, Self::Config>(
                acc,
                &mut out_unloader,
                CubeOption::new_None(),
                config.to_smm_config(),
                config,
            );
        }
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            !Self::is_consumer(config),
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            !Self::is_consumer(config),
            config,
        )
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

#[cube]
impl<
    MP: MatmulPrecision,
    SMM: StageMatmul<
            MP::ES,
            MP::EG,
            MP::EA,
            LhsReader = LhsBufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = RhsBufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> SpecializedMatmul<MP, SMM, LL, RL>
{
    fn is_consumer(#[comptime] config: <Self as GlobalMatmul<MP>>::Config) -> bool {
        UNIT_POS_Y < config.num_consumers()
    }
}
