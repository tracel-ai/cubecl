use crate::components::global::Specializer;
use crate::components::global::load::{StageBuffer, SyncPartialLoader, SyncPartialLoadingStrategy};
use crate::components::global::multi_stage::double_buffer_execution::{
    execute_current_and_load_next, execute_last_and_write_results, load_first,
};
use crate::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::components::stage::PartialStageToTileReader;
use crate::components::{LhsG, LhsS, MatmulIdent, RhsG, RhsS, global};
use crate::components::{MatmulPrecision, stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::div_ceil;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> global::GlobalMatmul<MP>
    for DoubleBufferingMatmul<MP, SMM, LL, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = PartialStageToTileReader<LhsS<MP>, LL::TilingLayout>,
            RhsReader = PartialStageToTileReader<RhsS<MP>, RL::TilingLayout>,
        >,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = SyncPartialLoader<MP::Lhs, Self::Config, LL>;
    type RhsLoader = SyncPartialLoader<MP::Rhs, Self::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let stage_step = config.tiling_scheme().elements_in_stage_k();
        let loop_step = stage_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = div_ceil(range, stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        SMM::zero_accumulator(acc, config.stage_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());

        let lhs_reader_a = Self::LhsLoader::reader(&lhs_loader, StageBuffer::A);
        let lhs_reader_b = Self::LhsLoader::reader(&lhs_loader, StageBuffer::B);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader, StageBuffer::A);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader, StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        load_first::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            StageBuffer::A,
            config,
        );

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader_a,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                StageBuffer::B,
                config,
            );

            // We always advance by 2 * k because stage B shares the same global memory state as stage A,
            // but it is implicitly offset by one stage's worth (k elements) when reading.
            Self::LhsLoader::advance_view(&mut lhs_loader, loop_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, loop_step);

            sync_cube();

            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader_b,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                StageBuffer::A,
                config,
            );

            sync_cube();
        }

        execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
            &lhs_reader_a,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            StageBuffer::B,
            config,
        );

        sync_cube();

        execute_last_and_write_results::<MP, SMM, Self::Config>(
            &lhs_reader_b,
            &rhs_reader_b,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut out_writer,
            &specializer,
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        SyncPartialLoader::<MP::Lhs, Self::Config, LL>::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        SyncPartialLoader::<MP::Rhs, Self::Config, RL>::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SMM::init_writer(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
