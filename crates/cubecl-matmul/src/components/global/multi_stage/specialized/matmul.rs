use crate::components::global::RoleRule;
use crate::components::global::read::SyncStrategy;
use crate::components::global::{GlobalConfig, GlobalWriter};
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
use cubecl_core::prelude::{barrier::BarrierLevel, *};
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
    for SpecializedMatmul<MP, SMM, LL, RL, GW>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy<SyncStrategy: SyncStrategy<Barrier = Barrier>>,
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
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let stage_step = config.tiling_scheme().elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = range.div_ceil(stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = num_stage_matmuls / 2;

        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let stage_bytes = comptime! {
            let lhs_bytes = config.stage_memory_config(MatmulIdent::Lhs).elements_in_stage() * lhs_elem_size;
            let rhs_bytes = config.stage_memory_config(MatmulIdent::Rhs).elements_in_stage() * rhs_elem_size;
            lhs_bytes + rhs_bytes
        };

        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_stage_a = lhs_reader.stage(StageBuffer::A);
        let lhs_stage_b = lhs_reader.stage(StageBuffer::B);
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let compute_units = config.plane_role_config().plane_roles.main_flow * config.plane_dim();

        let role_rule = RoleRule::new(config.role_rule_config());

        // Barrier for writing out
        let barrier_done = Barrier::new(BarrierLevel::cube_manual());

        // Barriers for releasing smem after compute
        let barrier_empty_a = Barrier::new(BarrierLevel::cube_manual());
        let barrier_empty_b = Barrier::new(BarrierLevel::cube_manual());

        // Barriers for marking smem as loaded
        let mut barrier_full_a = Barrier::new(BarrierLevel::cube_manual());
        let mut barrier_full_b = Barrier::new(BarrierLevel::cube_manual());

        if role_rule.elect_load_leader() {
            barrier_done.init_manual(compute_units);

            barrier_empty_a.init_manual(compute_units);
            barrier_empty_b.init_manual(compute_units);

            barrier_full_a.init_manual(1);
            barrier_full_b.init_manual(1);
            sync_async_proxy_shared();
        }
        sync_cube();

        let mut phase = 0;

        if role_rule.elect_load_leader() {
            for _ in 0..num_loops {
                barrier_empty_a.wait_parity(phase ^ 1);
                lhs_reader.load_stage(&mut barrier_full_a, StageBuffer::A, config);
                rhs_reader.load_stage(&mut barrier_full_a, StageBuffer::A, config);
                barrier_full_a.arrive_and_expect_tx(1, stage_bytes);

                barrier_empty_b.wait_parity(phase ^ 1);
                lhs_reader.load_stage(&mut barrier_full_b, StageBuffer::B, config);
                rhs_reader.load_stage(&mut barrier_full_b, StageBuffer::B, config);
                barrier_full_b.arrive_and_expect_tx(1, stage_bytes);

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

            SMM::write_results::<Self::GlobalWriter, Self::Config>(
                &acc,
                &mut out_stage,
                &mut out_writer,
                &partition_scheduler,
                config.stage_config(),
                config,
            );
        }
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
