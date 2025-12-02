use crate::components::global::multi_stage::double_buffer_execution::{
    execute_current_and_read_next, execute_last_and_write_results, read_first,
};
use crate::components::global::{GlobalMatmul, GlobalWriter, SharedGlobalMatmulConfig};
use crate::components::global::{Specializer, read::SyncStrategy};
use crate::components::stage::{FilledStage, StridedStageMemory};
use crate::components::stage::{StageConfig, StridedStageFamily};
use crate::components::{
    AccG,
    global::read::{
        PartialLoadingStrategy, PartialStageGlobalReader, StageBuffer, ZeroGlobalReader,
    },
};
use crate::components::{AccS, LhsG, LhsS, MatrixPrecision, RhsG, RhsS};
use crate::components::{MatmulPrecision, stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: PartialLoadingStrategy,
    RL: PartialLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL, GW> GlobalMatmul<MP>
    for DoubleBufferingMatmul<MP, SMM, LL, RL, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = StridedStageMemory<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStageMemory<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy<Stage = StridedStageFamily>,
    RL: PartialLoadingStrategy<Stage = StridedStageFamily, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    type LhsGlobalReader = PartialStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        LL,
    >;
    type RhsGlobalReader = PartialStageGlobalReader<
        <MP::Rhs as MatrixPrecision>::Global,
        <MP::Rhs as MatrixPrecision>::Stage,
        RL,
    >;
    type AccGlobalReader = ZeroGlobalReader<MP::Acc>;

    type GlobalWriter = GW;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let stage_step = config.stage_config.elements_in_stage_k();

        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        let mut acc = SMM::init_accumulators(config.stage_config);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        SMM::load_accumulators(&acc_reader.stage(), &mut acc, config.stage_config);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config);

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let mut barrier_a = LL::SyncStrategy::create_barrier();
        let mut barrier_b = LL::SyncStrategy::create_barrier();

        let specializer = Specializer::new(
            config.plane_role_config(),
            config.specialized_loading_sides(),
        );

        read_first::<LL::SyncStrategy, Self::LhsGlobalReader, Self::RhsGlobalReader>(
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_a,
            &specializer,
            StageBuffer::A,
            config.lhs_reader_config,
            config.rhs_reader_config,
        );

        LL::SyncStrategy::sync::<MP, _>(&mut barrier_a, config);

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SMM,
                LL::SyncStrategy,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_a,
                &rhs_stage_a,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier_b,
                &specializer,
                &partition_scheduler,
                StageBuffer::B,
                config,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();

            LL::SyncStrategy::sync::<MP, _>(&mut barrier_b, config);

            execute_current_and_read_next::<
                MP,
                SMM,
                LL::SyncStrategy,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage_b,
                &rhs_stage_b,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier_a,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            LL::SyncStrategy::sync::<MP, _>(&mut barrier_a, config);
        }

        execute_current_and_read_next::<
            MP,
            SMM,
            LL::SyncStrategy,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_stage_a,
            &rhs_stage_a,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_b,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        LL::SyncStrategy::sync::<MP, _>(&mut barrier_b, config);

        execute_last_and_write_results::<MP, GW, SMM, Self::Config>(
            &lhs_stage_b,
            &rhs_stage_b,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut out_writer,
            &specializer,
            &partition_scheduler,
            config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::<
            <MP::Lhs as MatrixPrecision>::Global,
            <MP::Lhs as MatrixPrecision>::Stage,
            LL,
        >::new(lhs, k_step, config.lhs_reader_config)
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        // We always advance by 2 * k because stage B shares the same global memory state as stage A,
        // but it is implicitly offset by one stage's worth (k elements) when reading.
        let k_step = config.stage_config.elements_in_stage_k() * 2;
        PartialStageGlobalReader::<
            <MP::Rhs as MatrixPrecision>::Global,
            <MP::Rhs as MatrixPrecision>::Stage,
            RL,
        >::new(rhs, k_step, config.rhs_reader_config)
    }

    fn init_acc_global_reader(
        acc: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        #[comptime] _config: Self::Config,
    ) -> Self::AccGlobalReader {
        match acc {
            CubeOption::None => ZeroGlobalReader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        Self::GlobalWriter::init(out, config.writer_config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config)
    }
}
