use crate::components::global::SharedGlobalMatmulConfig;
use crate::components::global::read::LoaderStage;
use crate::components::global::{GlobalConfig, GlobalWriter};
use crate::components::global::{RoleRule, read::AsyncPartialLoadingStrategy};
use crate::components::stage::FilledStage;
use crate::components::stage::StageConfig as _;
use crate::components::{
    AccG,
    global::read::{PartialStageGlobalReader, StageBuffer, ZeroGlobalReader},
};
use crate::components::{AccS, LhsG, LhsS, MatrixPrecision, RhsG, RhsS, global};
use crate::components::{MatmulPrecision, stage};

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords2d},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
/// Specializes planes to either read or compute planes.
/// Hardcoded for TMA right now
pub struct SpecializedMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    L: AsyncPartialLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _loading: PhantomData<L>,
    _writer: PhantomData<GW>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, L, GW> global::GlobalMatmul<MP> for SpecializedMatmul<MP, SMM, L, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = LoaderStage<L, LhsS<MP>>,
            RhsStage = LoaderStage<L, RhsS<MP>>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    L: AsyncPartialLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SharedGlobalMatmulConfig<SMM::Config>;

    type LhsGlobalReader = PartialStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        L,
    >;
    type RhsGlobalReader = PartialStageGlobalReader<
        <MP::Rhs as MatrixPrecision>::Global,
        <MP::Rhs as MatrixPrecision>::Stage,
        L,
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

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = num_stage_matmuls / 2;

        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let compute_units = config.plane_role_config().plane_roles.main_flow * config.plane_dim();

        let role_rule = RoleRule::new(config.plane_role_config().rule);

        // Barrier for writing out
        let barrier_done = Barrier::shared_uninit();

        // Barriers for releasing smem after compute
        let barrier_empty_a = Barrier::shared_uninit();
        let barrier_empty_b = Barrier::shared_uninit();

        // Barriers for marking smem as loaded
        let mut barrier_full_a = Barrier::shared_uninit();
        let mut barrier_full_b = Barrier::shared_uninit();

        if role_rule.elect_load_leader() {
            barrier_done.init_manual(compute_units);

            barrier_empty_a.init_manual(compute_units);
            barrier_empty_b.init_manual(compute_units);

            barrier_full_a.init_manual(L::arrival_count(config));
            barrier_full_b.init_manual(L::arrival_count(config));

            L::barrier_post_init();
        }
        sync_cube();

        let mut phase = 0;

        if L::is_elected(config) {
            for _ in 0..num_loops {
                barrier_empty_a.wait_parity(phase ^ 1);
                lhs_reader.load_stage(
                    &mut barrier_full_a,
                    StageBuffer::A,
                    config.lhs_reader_config,
                );
                rhs_reader.load_stage(
                    &mut barrier_full_a,
                    StageBuffer::A,
                    config.rhs_reader_config,
                );
                L::arrive::<MP, _>(&mut barrier_full_a, config);

                barrier_empty_b.wait_parity(phase ^ 1);
                lhs_reader.load_stage(
                    &mut barrier_full_b,
                    StageBuffer::B,
                    config.lhs_reader_config,
                );
                rhs_reader.load_stage(
                    &mut barrier_full_b,
                    StageBuffer::B,
                    config.rhs_reader_config,
                );
                L::arrive::<MP, _>(&mut barrier_full_b, config);

                lhs_reader.advance_view();
                rhs_reader.advance_view();
                phase ^= 1;
            }
        } else if role_rule.is_compute_plane() {
            let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
            let mut acc = SMM::init_accumulators(config.stage_config());

            SMM::load_accumulators(&acc_reader.stage(), &mut acc, config.stage_config());

            for _ in 0..num_loops {
                barrier_full_a.wait_parity(phase);
                SMM::execute(
                    &lhs_stage_a,
                    &rhs_stage_a,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    &mut acc,
                    config.stage_config(),
                    &partition_scheduler,
                );
                barrier_empty_a.arrive();

                barrier_full_b.wait_parity(phase);
                SMM::execute(
                    &lhs_stage_b,
                    &rhs_stage_b,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    &mut acc,
                    config.stage_config(),
                    &partition_scheduler,
                );
                barrier_empty_b.arrive();

                phase ^= 1;
            }
            barrier_done.arrive_and_wait();

            lhs_reader.free_stage();
            rhs_reader.free_stage();

            let mut out_stage = Self::GlobalWriter::stage(&out_writer);

            SMM::write_results::<Self::GlobalWriter>(
                &acc,
                &mut out_stage,
                &mut out_writer,
                &partition_scheduler,
                config.stage_config(),
            );
        }
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
            L,
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
            L,
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
        SMM::init_accumulators(config.stage_config())
    }
}
