use std::marker::PhantomData;

use crate::components::MatmulPrecision;
use crate::components::RhsG;
use crate::components::RhsS;
use crate::components::global::GlobalConfig;
use crate::components::global::GlobalMatmul;
use crate::components::global::memory::SimpleGlobalLayout;
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
use barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords2d};

/// Performs matrix multiplication at the global level
/// Similar to simple matmul but using asynchronous loading
pub struct SimpleBarrierMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> GlobalMatmul<MP> for SimpleBarrierMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
            MP,
            LhsStage = StridedStage<LhsS<MP>, LL::TilingLayout>,
            RhsStage = StridedStage<RhsS<MP>, RL::TilingLayout>,
            AccStage = FilledStage<AccS<MP>>,
            WriteCoords = Coords2d,
        >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Config = SimpleBarrierConfig<SMM::Config>;
    type LhsGlobalReader =
        AsyncFullStageGlobalReader<MP::Lhs, Barrier, SMM::Config, LL, Self::Config>;
    type RhsGlobalReader =
        AsyncFullStageGlobalReader<MP::Rhs, Barrier, SMM::Config, RL, Self::Config>;
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
            lhs.view(layout).slice(offset, slice_size),
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
            rhs.view(layout).slice(offset, slice_size),
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
