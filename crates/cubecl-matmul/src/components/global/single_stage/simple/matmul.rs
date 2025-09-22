use crate::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalMatmul,
        load::{SyncFullLoadingStrategy, SyncFullStageLoader, ZeroStageLoader},
        memory::SimpleGlobalLayout,
        single_stage::simple::SimpleConfig,
    },
    stage::{FillStageReader, FullStageReader, StageMatmul},
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
            LhsStageReader = FullStageReader<LhsS<MP>, LL::TilingLayout>,
            RhsStageReader = FullStageReader<RhsS<MP>, RL::TilingLayout>,
            AccStageReader = FillStageReader<AccS<MP>>,
            WriteCoords = Coords2d,
        >,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Config = SimpleConfig<SMM::Config>;
    type LhsStageLoader = SyncFullStageLoader<MP::Lhs, Self::Config, LL>;
    type RhsStageLoader = SyncFullStageLoader<MP::Rhs, Self::Config, RL>;
    type AccStageLoader = ZeroStageLoader<MP::Acc>;
    type StageUnloader = SMM::StageUnloader;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_loader: Self::LhsStageLoader,
        mut rhs_loader: Self::RhsStageLoader,
        acc_loader: Self::AccStageLoader,
        mut out_writer: Self::StageUnloader,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        SMM::load_accumulators(&acc_loader.reader(), acc, config.stage_config());

        let lhs_stage_reader = &lhs_loader.reader();
        let rhs_stage_reader = &rhs_loader.reader();

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsStageLoader::load_stage(&mut lhs_loader, config);
            Self::RhsStageLoader::load_stage(&mut rhs_loader, config);

            sync_cube();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            Self::LhsStageLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsStageLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::write_results::<Self::Config>(
            acc,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
            config,
        );
    }

    fn init_lhs_stage_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let conf = config.global_memory_config(MatmulIdent::Lhs);
        let layout = SimpleGlobalLayout::new(&lhs, batch_offset, conf);
        Self::LhsStageLoader::new(
            lhs.view(layout).slice_unchecked(offset, slice_size),
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_stage_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        let conf = config.global_memory_config(MatmulIdent::Rhs);
        let layout = SimpleGlobalLayout::new(&rhs, batch_offset, conf);
        Self::RhsStageLoader::new(
            rhs.view(layout).slice_unchecked(offset, slice_size),
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_acc_stage_loader(
        acc: CubeOption<VirtualTensor<AccG<MP>>>,
        _batch_offset: u32,
        _offset: Coords2d,
        _slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] _config: Self::Config,
    ) -> Self::AccStageLoader {
        match acc {
            CubeOption::None => ZeroStageLoader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let conf = config.global_memory_config(MatmulIdent::Out);
        let layout = SimpleGlobalLayout::new(&out, batch_offset, conf);
        SMM::init_writer(out.view_mut(layout).slice_mut_unchecked(offset, slice_size))
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
