use crate::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalMatmul,
        memory::SimpleGlobalLayout,
        read::{SyncFullLoadingStrategy, SyncFullStageGlobalReader, ZeroGlobalReader},
        single_stage::simple::SimpleConfig,
    },
    stage::{FilledStage, StageMatmul, StridedStage},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{layout::Coords2d, r#virtual::VirtualTensor},
};
use std::marker::PhantomData;

use crate::components::global::GlobalConfig;

/// Performs matrix multiplication at the global level.
///
/// Fully loads all stages, synchronizes all planes, performs computation,
/// synchronizes again, then proceeds to the next set of stages.
pub struct SimpleMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
> {
    _phantom: PhantomData<(MP, SMM, LL, RL)>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> GlobalMatmul<MP> for SimpleMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
            MP,
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
        >,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Config = SimpleConfig<SMM::Config>;
    type LhsGlobalReader = SyncFullStageGlobalReader<MP::Lhs, Self::Config, LL>;
    type RhsGlobalReader = SyncFullStageGlobalReader<MP::Rhs, Self::Config, RL>;
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
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let lhs_stage = &lhs_reader.stage();
        let rhs_stage = &rhs_reader.stage();

        for _ in 0..num_loops {
            sync_cube();

            lhs_reader.load_stage(config);
            rhs_reader.load_stage(config);

            sync_cube();

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

        SMM::write_results::<Self::Config>(
            acc,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
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
        let layout = SimpleGlobalLayout::new(&lhs, batch_offset, conf);
        Self::LhsGlobalReader::new(
            lhs.view(layout).slice_unchecked(offset, slice_size),
            config.k_step,
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
        let layout = SimpleGlobalLayout::new(&rhs, batch_offset, conf);
        Self::RhsGlobalReader::new(
            rhs.view(layout).slice_unchecked(offset, slice_size),
            config.k_step,
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
        SMM::init_writer(out.view_mut(layout).slice_mut_unchecked(offset, size), conf)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
