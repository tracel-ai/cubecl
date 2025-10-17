use std::marker::PhantomData;

use crate::components::RhsG;
use crate::components::RhsS;
use crate::components::global::GlobalConfig;
use crate::components::global::GlobalMatmul;
use crate::components::global::read::AsyncFullLoadingStrategy;
use crate::components::global::read::AsyncFullStageGlobalReader;
use crate::components::global::single_stage::barrier::SimpleBarrierConfig;
use crate::components::stage::StageMatmul;
use crate::components::{AccG, AccS, LhsS};
use crate::components::{LhsG, global::read::ZeroGlobalReader};
use crate::components::{
    MatmulIdent,
    stage::{FilledStage, StridedStage},
};
use crate::components::{MatmulPrecision, global::GlobalWriter};
use barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::View;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords2d};

/// Performs matrix multiplication at the global level
/// Similar to simple matmul but using asynchronous loading
pub struct SimpleBarrierMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
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
    for SimpleBarrierMatmul<MP, SMM, LL, RL, GW>
where
    SMM: StageMatmul<
            MP,
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SimpleBarrierConfig<SMM::Config>;
    type LhsGlobalReader =
        AsyncFullStageGlobalReader<MP::Lhs, Barrier, SMM::Config, LL, Self::Config>;
    type RhsGlobalReader =
        AsyncFullStageGlobalReader<MP::Rhs, Barrier, SMM::Config, RL, Self::Config>;
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
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let acc_reader = acc_reader.stage();
        SMM::load_accumulators(&acc_reader, acc, config.stage_config());

        let barrier_level = LL::barrier_level();
        let lhs_barrier = Barrier::new(barrier_level);
        let rhs_barrier = Barrier::new(barrier_level);

        for loop_iter in 0..num_loops {
            sync_cube();

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    lhs_reader.clear_stage(config);
                    rhs_reader.clear_stage(config);
                    sync_cube();
                }
            }

            // Start loading
            lhs_reader.load_stage(&lhs_barrier, config);
            rhs_reader.load_stage(&rhs_barrier, config);

            let lhs_stage = &lhs_reader.stage();
            let rhs_stage = &rhs_reader.stage();

            lhs_barrier.arrive_and_wait();
            rhs_barrier.arrive_and_wait();

            SMM::execute(
                lhs_stage,
                rhs_stage,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();
        }

        let mut out_stage = Self::GlobalWriter::stage(&out_writer);

        SMM::write_results::<Self::GlobalWriter, Self::Config>(
            acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
            config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        Self::LhsGlobalReader::new(lhs, config.k_step, MatmulIdent::Lhs, config)
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(rhs, config.k_step, MatmulIdent::Rhs, config)
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
