use crate::components::global::multi_stage::double_buffer_execution::{
    execute_current_and_read_next, execute_last_and_write_results, read_first,
};
use crate::components::global::{GlobalConfig, GlobalWriter};
use crate::components::global::{Specializer, read::SyncStrategy};
use crate::components::{
    AccG,
    global::read::{
        PartialLoadingStrategy, PartialStageGlobalReader, StageBuffer, ZeroGlobalReader,
    },
};
use crate::components::{AccS, LhsG, LhsS, MatmulIdent, RhsG, RhsS, global};
use crate::components::{MatmulPrecision, stage};
use crate::components::{
    global::multi_stage::double_buffering::DoubleBufferingGlobalConfig,
    stage::{FilledStage, StridedStage},
};
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
impl<MP: MatmulPrecision, SMM, LL, RL, GW> global::GlobalMatmul<MP>
    for DoubleBufferingMatmul<MP, SMM, LL, RL, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy,
    RL: PartialLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    type LhsGlobalReader = PartialStageGlobalReader<MP::Lhs, Self::Config, LL>;
    type RhsGlobalReader = PartialStageGlobalReader<MP::Rhs, Self::Config, RL>;
    type AccGlobalReader = ZeroGlobalReader<MP::Acc>;

    type GlobalWriter = GW;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let stage_step = config.tiling_scheme().elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let mut barrier_a = LL::SyncStrategy::create_barrier();
        let mut barrier_b = LL::SyncStrategy::create_barrier();

        let specializer = Specializer::new::<Self::Config>(config);

        read_first::<
            MP,
            SMM,
            LL::SyncStrategy,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_a,
            &specializer,
            StageBuffer::A,
            config,
        );

        LL::SyncStrategy::sync::<MP, Self::Config>(&mut barrier_a, config);

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
                acc,
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

            LL::SyncStrategy::sync::<MP, Self::Config>(&mut barrier_b, config);

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
                acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier_a,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            LL::SyncStrategy::sync::<MP, Self::Config>(&mut barrier_a, config);
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
            acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier_b,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        LL::SyncStrategy::sync::<MP, Self::Config>(&mut barrier_b, config);

        execute_last_and_write_results::<MP, GW, SMM, Self::Config>(
            &lhs_stage_b,
            &rhs_stage_b,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
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
        let k_step = k_step::<Self::Config>(config);
        PartialStageGlobalReader::<MP::Lhs, Self::Config, LL>::new(
            lhs,
            k_step,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        let k_step = k_step::<Self::Config>(config);
        PartialStageGlobalReader::<MP::Rhs, Self::Config, RL>::new(
            rhs,
            k_step,
            MatmulIdent::Rhs,
            config,
        )
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
        let conf = config.global_memory_config(MatmulIdent::Out);
        Self::GlobalWriter::init::<SMM::Config>(out, conf, config.stage_config())
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}

/// We always advance by 2 * k because stage B shares the same global memory state as stage A,
/// but it is implicitly offset by one stage's worth (k elements) when reading.
#[cube]
fn k_step<C: GlobalConfig>(#[comptime] config: C) -> u32 {
    let step = config.tiling_scheme().elements_in_stage_k() * 2;
    step.runtime()
}
