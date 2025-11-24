use crate::components::global::{self, GlobalWriter, SharedGlobalMatmulConfig};
use crate::components::global::{Specializer, read::sync::Synchronous};
use crate::components::stage::StageConfig as _;
use crate::components::stage::StridedStageFamily;
use crate::components::{
    AccG,
    global::read::{
        FullLoadingStrategy, FullStageGlobalReader, PartialLoadingStrategy,
        PartialStageGlobalReader, StageBuffer, ZeroGlobalReader,
    },
};
use crate::components::{AccS, global::multi_stage::ordered::LL};
use crate::components::{LhsG, LhsS, MatmulPrecision, MatrixPrecision, RhsG, RhsS, stage};
use crate::components::{
    global::multi_stage::double_buffer_execution::{
        execute_current_and_read_next, execute_last_and_write_results, read_first,
    },
    stage::{FilledStage, StridedStageMemory},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level.
/// Uses double buffering with two shared memory buffers for `Rhs`,
/// but only one for `Lhs`—the second "buffer" for `Lhs` is the fragments themselves.
/// For this to work, the `Lhs` reader planes must compute using
/// only the data they have loaded themselves.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
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
impl<MP: MatmulPrecision, SMM, RL, GW> global::GlobalMatmul<MP>
    for OrderedDoubleBufferingMatmul<MP, SMM, RL, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = StridedStageMemory<LhsS<MP>, <LL as FullLoadingStrategy>::TilingLayout>,
            RhsStage = StridedStageMemory<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    RL: PartialLoadingStrategy<Stage = StridedStageFamily, SyncStrategy = Synchronous>,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;
    type LhsGlobalReader = FullStageGlobalReader<
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

        let acc_reader = acc_reader.stage();
        SMM::load_accumulators(&acc_reader, &mut acc, config.stage_config);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config);

        let lhs_stage = lhs_reader.stage();
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let mut barrier = ();

        let specializer = Specializer::new(
            config.plane_role_config(),
            config.specialized_loading_sides(),
        );

        read_first::<Synchronous, Self::LhsGlobalReader, Self::RhsGlobalReader>(
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier,
            &specializer,
            StageBuffer::A,
            config.lhs_reader_config,
            config.rhs_reader_config,
        );

        lhs_reader.advance_view();

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SMM,
                Synchronous,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage,
                &rhs_stage_a,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier,
                &specializer,
                &partition_scheduler,
                StageBuffer::B,
                config,
            );

            lhs_reader.advance_view();
            rhs_reader.advance_view();

            sync_cube();

            execute_current_and_read_next::<
                MP,
                SMM,
                Synchronous,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage,
                &rhs_stage_b,
                &mut lhs_tile,
                &mut rhs_tile,
                &mut acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &mut barrier,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            lhs_reader.advance_view();

            sync_cube();
        }

        execute_current_and_read_next::<
            MP,
            SMM,
            Synchronous,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_stage,
            &rhs_stage_a,
            &mut lhs_tile,
            &mut rhs_tile,
            &mut acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &mut barrier,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        sync_cube();

        execute_last_and_write_results::<MP, GW, SMM, Self::Config>(
            &lhs_stage,
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
        // We always advance by only k for Lhs
        let k_step = config.stage_config.elements_in_stage_k();
        FullStageGlobalReader::<
            <MP::Lhs as MatrixPrecision>::Global,
            <MP::Lhs as MatrixPrecision>::Stage,
            LL,
        >::new(lhs, k_step, config.lhs_reader_config)
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        // We always advance by 2 * k for Rhs only
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
