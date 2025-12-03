use crate::components::{
    AccG, AccS, LhsG, LhsS, MatmulPrecision, MatrixPrecision, RhsG, RhsS,
    global::{
        GlobalMatmul, GlobalWriter, SharedGlobalMatmulConfig,
        read::{FullLoadingStrategy, FullStageGlobalReader, SyncStrategy, ZeroGlobalReader},
    },
    stage::StridedStageMemory,
    stage::{FilledStage, StageConfig, StageMatmul},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level.
///
/// Fully loads all stages, synchronizes all planes, performs computation,
/// synchronizes again, then proceeds to the next set of stages.
pub struct SimpleMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
> {
    _phantom: PhantomData<(MP, SMM, LL, RL, GW)>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL, GW> GlobalMatmul<MP> for SimpleMatmul<MP, SMM, LL, RL, GW>
where
    SMM: StageMatmul<
            MP,
            LhsStage = StridedStageMemory<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStageMemory<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;
    type LhsGlobalReader = FullStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        LL,
    >;
    type RhsGlobalReader = FullStageGlobalReader<
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
        let k_step = config.stage_config.elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let mut acc = SMM::init_accumulators(config.stage_config);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config);

        SMM::load_accumulators(&acc_reader.stage(), &mut acc, config.stage_config);

        let lhs_stage = &lhs_reader.stage();
        let rhs_stage = &rhs_reader.stage();

        let mut barrier = LL::SyncStrategy::create_barrier();

        for i in 0..num_loops {
            sync_cube();

            #[allow(clippy::collapsible_if)]
            if comptime![(LL::SHOULD_CLEAR || RL::SHOULD_CLEAR) && config.check_k_bounds()] {
                if i == num_loops - 1 {
                    lhs_reader.clear_stage(config.lhs_reader_config);
                    rhs_reader.clear_stage(config.rhs_reader_config);
                }
            }

            lhs_reader.load_stage(&mut barrier, config.lhs_reader_config);
            rhs_reader.load_stage(&mut barrier, config.rhs_reader_config);

            LL::SyncStrategy::sync::<MP, _>(&mut barrier, config);

            SMM::execute(
                lhs_stage,
                rhs_stage,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                config.stage_config,
                &partition_scheduler,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();
        }

        // Frees input stages for reuse, so the output stage can be allocated into the same
        // range. The `sync_cube` is required to ensure other planes are done reading from the stages.
        //
        // This is currently very unintuitive, because while the stage already exists, it actually
        // isn't allocated until it's used (by writing to it). We should eventually separate the
        // write call into a different function and defer creating the writer until after the stages
        // are freed to make the order of operations more clear.
        sync_cube();
        lhs_reader.free_stage();
        rhs_reader.free_stage();

        let mut out_stage = Self::GlobalWriter::stage(&out_writer);

        SMM::write_results::<Self::GlobalWriter>(
            &acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        Self::LhsGlobalReader::new(
            lhs,
            config.stage_config.elements_in_stage_k(),
            config.lhs_reader_config,
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(
            rhs,
            config.stage_config.elements_in_stage_k(),
            config.rhs_reader_config,
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
        Self::GlobalWriter::init(out, config.writer_config)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config)
    }
}
