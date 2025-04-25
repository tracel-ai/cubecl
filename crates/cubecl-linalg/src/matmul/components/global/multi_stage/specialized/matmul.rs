use crate::matmul::components::{
    Ident, InputIdent, MatmulPrecision,
    global::{
        self, GlobalMatmul, Quantization, ZeroAccumulatorLoader,
        load::{BufferId, SyncBufferLoader, SyncBufferLoadingStrategy},
        output_loader::Unloader,
    },
    stage::{BufferReader, StageMatmul},
};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};

use super::config::Config;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};

use crate::matmul::{
    components::{
        InvalidConfigError, MatmulConfigFactory, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{self, BufferReaderFamily},
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
    SMM: stage::StageMatmulFamily<LhsReader = BufferReaderFamily, RhsReader = BufferReaderFamily>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SpecializedMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
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
    SMM: StageMatmul<MP>,
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
            MP,
            LhsReader = BufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = BufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = (
        SyncBufferLoader<MP, Self::Config, LL>,
        SyncBufferLoader<MP, Self::Config, LL>,
    );
    type RhsLoader = (
        SyncBufferLoader<MP, Self::Config, RL>,
        SyncBufferLoader<MP, Self::Config, RL>,
    );
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let is_consumer = Self::is_consumer(config);
        let is_producer = !is_consumer;

        let buffer_step = config.tiling_dimensions(Ident::Lhs).total_col();
        let loop_step = buffer_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stages = div_ceil(range, buffer_step);

        // Algorithm assumes an even number of stages
        let num_stages = needed_stages + (needed_stages % 2);
        let num_loops = num_stages / 2;

        SMM::zero_accumulator(acc, config.to_smm_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        let mut lhs_loader_a = lhs_loader.0;
        let mut lhs_loader_b = lhs_loader.1;
        let mut rhs_loader_a = rhs_loader.0;
        let mut rhs_loader_b = rhs_loader.1;

        let lhs_reader_a = SyncBufferLoader::<MP, Self::Config, LL>::reader(&lhs_loader_a);
        let lhs_reader_b = SyncBufferLoader::<MP, Self::Config, LL>::reader(&lhs_loader_b);
        let rhs_reader_a = SyncBufferLoader::<MP, Self::Config, RL>::reader(&rhs_loader_a);
        let rhs_reader_b = SyncBufferLoader::<MP, Self::Config, RL>::reader(&rhs_loader_b);

        SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_b, buffer_step);
        SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_b, buffer_step);

        for _ in 0..num_loops {
            if is_producer {
                SyncBufferLoader::<MP, Self::Config, LL>::fill_stage(&mut lhs_loader_a, config);
                SyncBufferLoader::<MP, Self::Config, RL>::fill_stage(&mut rhs_loader_a, config);
            }

            sync_units();

            if is_consumer {
                SMM::execute(
                    &lhs_reader_a,
                    &rhs_reader_a,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    acc,
                    config.to_smm_config(),
                );
            }

            SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_a, loop_step);
            SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_a, loop_step);

            if is_producer {
                SyncBufferLoader::<MP, Self::Config, LL>::fill_stage(&mut lhs_loader_b, config);
                SyncBufferLoader::<MP, Self::Config, RL>::fill_stage(&mut rhs_loader_b, config);
            }

            sync_units();

            if is_consumer {
                SMM::execute(
                    &lhs_reader_b,
                    &rhs_reader_b,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    acc,
                    config.to_smm_config(),
                );
            }

            SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_b, loop_step);
            SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_b, loop_step);
        }

        if is_consumer {
            SMM::read_accumulator::<Self::Out, Self::Config>(
                acc,
                &mut out_unloader,
                config.to_smm_config(),
                config,
            );
        }
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
        (
            SyncBufferLoader::<MP, Self::Config, LL>::new(
                lhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::A,
                InputIdent::Lhs,
                config,
            ),
            SyncBufferLoader::<MP, Self::Config, LL>::new(
                lhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::B,
                InputIdent::Lhs,
                config,
            ),
        )
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
        (
            SyncBufferLoader::<MP, Self::Config, RL>::new(
                rhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::A,
                InputIdent::Rhs,
                config,
            ),
            SyncBufferLoader::<MP, Self::Config, RL>::new(
                rhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::B,
                InputIdent::Rhs,
                config,
            ),
        )
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

#[cube]
impl<
    MP: MatmulPrecision,
    SMM: StageMatmul<
            MP,
            LhsReader = BufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = BufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> SpecializedMatmul<MP, SMM, LL, RL>
{
    fn is_consumer(#[comptime] config: <Self as GlobalMatmul<MP>>::Config) -> bool {
        UNIT_POS_Y < config.num_consumers()
    }
}
