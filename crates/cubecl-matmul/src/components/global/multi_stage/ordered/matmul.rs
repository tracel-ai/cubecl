use crate::components::global::{self, GlobalConfig, GlobalWriter};
use crate::components::global::{Specializer, memory::SimpleGlobalLayout};
use crate::components::{
    AccG,
    global::read::{
        StageBuffer, SyncFullLoadingStrategy, SyncFullStageGlobalReader,
        SyncPartialLoadingStrategy, SyncPartialStageGlobalReader, ZeroGlobalReader,
    },
};
use crate::components::{AccS, global::multi_stage::ordered::LL};
use crate::components::{LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS, stage};
use crate::components::{
    global::multi_stage::double_buffer_execution::{
        execute_current_and_read_next, execute_last_and_write_results, read_first,
    },
    stage::{FilledStage, StridedStage},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor};
use std::marker::PhantomData;

use super::OrderedDoubleBufferingGlobalConfig;

/// Performs matrix multiplication at the global level.
/// Uses double buffering with two shared memory buffers for `Rhs`,
/// but only one for `Lhs`—the second "buffer" for `Lhs` is the fragments themselves.
/// For this to work, the `Lhs` reader planes must compute using
/// only the data they have loaded themselves.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    RL: SyncPartialLoadingStrategy,
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
            LhsStage = StridedStage<LhsS<MP>, <LL as SyncFullLoadingStrategy>::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    RL: SyncPartialLoadingStrategy,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;
    type LhsGlobalReader = SyncFullStageGlobalReader<MP::Lhs, Self::Config, LL>;
    type RhsGlobalReader = SyncPartialStageGlobalReader<MP::Rhs, Self::Config, RL>;
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

        let acc_reader = acc_reader.stage();
        SMM::load_accumulators(&acc_reader, acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_stage = lhs_reader.stage();
        let rhs_stage_a = rhs_reader.stage(StageBuffer::A);
        let rhs_stage_b = rhs_reader.stage(StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        read_first::<MP, SMM, Self::LhsGlobalReader, Self::RhsGlobalReader, Self::Config>(
            &mut lhs_reader,
            &mut rhs_reader,
            &specializer,
            StageBuffer::A,
            config,
        );

        lhs_reader.advance_view();

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_read_next::<
                MP,
                SMM,
                Self::LhsGlobalReader,
                Self::RhsGlobalReader,
                Self::Config,
            >(
                &lhs_stage,
                &rhs_stage_a,
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
                &lhs_stage,
                &rhs_stage_b,
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

            lhs_reader.advance_view();

            sync_cube();
        }

        execute_current_and_read_next::<
            MP,
            SMM,
            Self::LhsGlobalReader,
            Self::RhsGlobalReader,
            Self::Config,
        >(
            &lhs_stage,
            &rhs_stage_a,
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

        execute_last_and_write_results::<MP, GW, SMM, Self::Config>(
            &lhs_stage,
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
        lhs: VirtualTensor<LhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        let conf = config.global_memory_config(MatmulIdent::Lhs);
        let k_step = lhs_k_step::<Self::Config>(config);
        let layout = SimpleGlobalLayout::new(&lhs, batch_offset, conf);
        SyncFullStageGlobalReader::<MP::Lhs, Self::Config, LL>::new(
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
        let k_step = rhs_k_step::<Self::Config>(config);
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
        let view = out.view_mut(layout).slice_mut_unchecked(offset, size);
        Self::GlobalWriter::init::<SMM::Config>(view, conf, config.stage_config())
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}

#[cube]
fn lhs_k_step<C: GlobalConfig>(#[comptime] config: C) -> u32 {
    let step = config.tiling_scheme().elements_in_stage_k();
    step.runtime()
}
#[cube]
fn rhs_k_step<C: GlobalConfig>(#[comptime] config: C) -> u32 {
    let step = config.tiling_scheme().elements_in_stage_k() * 2;
    step.runtime()
}
