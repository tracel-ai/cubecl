use crate::components::global::{GlobalConfig, GlobalWriter};
use crate::components::global::{
    PartitionedStage,
    multi_stage::double_buffer_execution::{
        execute_current_and_read_next, execute_last_and_write_results, read_first,
    },
};
use crate::components::global::{Specializer, memory::SimpleGlobalLayout};
use crate::components::{
    AccG,
    global::read::{
        StageBuffer, SyncPartialLoadingStrategy, SyncPartialStageGlobalReader, ZeroGlobalReader,
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
    tensor::{layout::Coords2d, r#virtual::VirtualTensor},
};
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on stage A,
/// they trigger a computation event from tensor cores on stage B. Then stages are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
> where
    SMM::GlobalWriter: GlobalWriter<MP::Acc>,
{
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
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = PartitionedStage<AccS<MP>>,
        >,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    type LhsGlobalReader = SyncPartialStageGlobalReader<MP::Lhs, Self::Config, LL>;
    type RhsGlobalReader = SyncPartialStageGlobalReader<MP::Rhs, Self::Config, RL>;
    type AccGlobalReader = ZeroGlobalReader<MP::Acc>;

    type GlobalWriter = SMM::GlobalWriter;
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

        let lhs_reader_a = lhs_reader.stage(StageBuffer::A);
        let lhs_reader_b = lhs_reader.stage(StageBuffer::B);
        let rhs_reader_a = rhs_reader.stage(StageBuffer::A);
        let rhs_reader_b = rhs_reader.stage(StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        read_first::<MP, SMM, Self::LhsGlobalReader, Self::RhsGlobalReader, Self::Config>(
            &mut lhs_reader,
            &mut rhs_reader,
            &specializer,
            StageBuffer::A,
            config,
        );

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SMM,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_reader_a,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_reader,
                &mut rhs_reader,
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
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_reader_b,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_reader,
                &mut rhs_reader,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            sync_cube();
        }

        execute_current_and_read_next::<
            MP,
            SMM,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_reader_a,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut lhs_reader,
            &mut rhs_reader,
            &specializer,
            &partition_scheduler,
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
            &partition_scheduler,
            config,
        );
    }

    fn init_lhs_global_reader(
        lhs: VirtualTensor<LhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        let conf = config.global_memory_config(MatmulIdent::Lhs);
        let k_step = k_step::<Self::Config>(config);
        let layout = SimpleGlobalLayout::new(&lhs, batch_offset, conf);
        SyncPartialStageGlobalReader::<MP::Lhs, Self::Config, LL>::new(
            lhs.view(layout).slice_unchecked(offset, slice_size),
            k_step,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_global_reader(
        rhs: VirtualTensor<RhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        let conf = config.global_memory_config(MatmulIdent::Rhs);
        let k_step = k_step::<Self::Config>(config);
        let layout = SimpleGlobalLayout::new(&rhs, batch_offset, conf);
        SyncPartialStageGlobalReader::<MP::Rhs, Self::Config, RL>::new(
            rhs.view(layout).slice_unchecked(offset, slice_size),
            k_step,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_acc_global_reader(
        acc: CubeOption<VirtualTensor<AccG<MP>>>,
        _batch_offset: u32,
        _offset: Coords2d,
        _slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] _config: Self::Config,
    ) -> Self::AccGlobalReader {
        match acc {
            CubeOption::None => ZeroGlobalReader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        batch_offset: u32,
        offset: Coords2d,
        size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        let conf = config.global_memory_config(MatmulIdent::Out);
        let layout = SimpleGlobalLayout::new(&out, batch_offset, conf);
        SMM::init_writer(
            out.view_mut(layout).slice_mut_unchecked(offset, size),
            conf,
            config.stage_config,
        )
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
