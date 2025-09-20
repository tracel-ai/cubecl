use crate::components::global::{self, GlobalConfig};
use crate::components::global::{Specializer, memory::SimpleGlobalLayout};
use crate::components::stage::FullStageReader;
use crate::components::stage::PartialStageReader;
use crate::components::{
    AccG,
    global::load::{
        StageBuffer, SyncFullLoadingStrategy, SyncFullStageLoader, SyncPartialLoadingStrategy,
        SyncPartialStageLoader, ZeroStageLoader,
    },
};
use crate::components::{AccS, global::multi_stage::ordered::LL};
use crate::components::{LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS, stage};
use crate::components::{
    global::multi_stage::double_buffer_execution::{
        execute_current_and_load_next, execute_last_and_write_results, load_first,
    },
    stage::FillStageReader,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor};
use cubecl_std::{div_ceil, tensor::layout::Coords2d};
use std::marker::PhantomData;

use super::OrderedDoubleBufferingGlobalConfig;

/// Performs matrix multiplication at the global level.
/// Uses double buffering with two shared memory buffers for `Rhs`,
/// but only one for `Lhs`—the second "buffer" for `Lhs` is the fragments themselves.
/// For this to work, the `Lhs` loader planes must compute using
/// only the data they have loaded themselves.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    RL: SyncPartialLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, RL> global::GlobalMatmul<MP>
    for OrderedDoubleBufferingMatmul<MP, SMM, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsStageReader = FullStageReader<
                LhsS<MP>,
                <LL as SyncFullLoadingStrategy>::TilingLayout,
            >,
            RhsStageReader = PartialStageReader<RhsS<MP>, RL::TilingLayout>,
            AccStageReader = FillStageReader<AccS<MP>>,
            WriteCoords = Coords2d,
        >,
    RL: SyncPartialLoadingStrategy,
{
    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;
    type LhsStageLoader = SyncFullStageLoader<MP::Lhs, Self::Config, LL>;
    type RhsStageLoader = SyncPartialStageLoader<MP::Rhs, Self::Config, RL>;
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
        let stage_step = config.tiling_scheme().elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = div_ceil(range, stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        let acc_reader = acc_loader.reader();
        SMM::load_accumulators(&acc_reader, acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_reader = lhs_loader.reader();
        let rhs_reader_a = rhs_loader.reader(StageBuffer::A);
        let rhs_reader_b = rhs_loader.reader(StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        load_first::<MP, SMM, Self::LhsStageLoader, Self::RhsStageLoader, Self::Config>(
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            StageBuffer::A,
            config,
        );

        lhs_loader.advance_view();

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_load_next::<
                MP,
                SMM,
                Self::LhsStageLoader,
                Self::RhsStageLoader,
                Self::Config,
            >(
                &lhs_reader,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                &partition_scheduler,
                StageBuffer::B,
                config,
            );

            lhs_loader.advance_view();
            rhs_loader.advance_view();

            sync_cube();

            execute_current_and_load_next::<
                MP,
                SMM,
                Self::LhsStageLoader,
                Self::RhsStageLoader,
                Self::Config,
            >(
                &lhs_reader,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                &partition_scheduler,
                StageBuffer::A,
                config,
            );

            lhs_loader.advance_view();

            sync_cube();
        }

        execute_current_and_load_next::<
            MP,
            SMM,
            Self::LhsStageLoader,
            Self::RhsStageLoader,
            Self::Config,
        >(
            &lhs_reader,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            &partition_scheduler,
            StageBuffer::B,
            config,
        );

        sync_cube();

        execute_last_and_write_results::<MP, SMM, Self::Config>(
            &lhs_reader,
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

    fn init_lhs_stage_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        batch_offset: u32,
        offset: Coords2d,
        slice_size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let conf = config.global_memory_config(MatmulIdent::Lhs);
        let k_step = lhs_k_step::<Self::Config>(config);
        let layout = SimpleGlobalLayout::new(&lhs, batch_offset, conf);
        SyncFullStageLoader::<MP::Lhs, Self::Config, LL>::new(
            lhs.view(layout).slice_unchecked(offset, slice_size),
            k_step,
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
        let k_step = rhs_k_step::<Self::Config>(config);
        let layout = SimpleGlobalLayout::new(&rhs, batch_offset, conf);
        SyncPartialStageLoader::<MP::Rhs, Self::Config, RL>::new(
            rhs.view(layout).slice_unchecked(offset, slice_size),
            k_step,
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
        size: Coords2d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let conf = config.global_memory_config(MatmulIdent::Out);
        let layout = SimpleGlobalLayout::new(&out, batch_offset, conf);
        SMM::init_writer(out.view_mut(layout).slice_mut_unchecked(offset, size), conf)
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
