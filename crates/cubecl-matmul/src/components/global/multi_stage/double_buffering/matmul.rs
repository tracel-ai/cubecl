use crate::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::components::global::{GlobalConfig, StageUnloader};
use crate::components::global::{Specializer, memory::SimpleGlobalLayout};
use crate::components::stage::PartialStageReader;
use crate::components::{
    AccG,
    global::load::{
        StageBuffer, SyncPartialLoadingStrategy, SyncPartialStageLoader, ZeroStageLoader,
    },
};
use crate::components::{AccS, LhsG, LhsS, MatmulIdent, RhsG, RhsS, global};
use crate::components::{MatmulPrecision, stage};
use crate::components::{
    global::multi_stage::double_buffer_execution::{
        execute_current_and_load_next, execute_last_and_write_results, load_first,
    },
    stage::FillStageReader,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor};
use cubecl_std::{div_ceil, tensor::layout::Coords3d};
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
    SMM::StageUnloader: StageUnloader<AccG<MP>, Coordinates = Coords3d>,
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
            LhsStageReader = PartialStageReader<LhsS<MP>, LL::TilingLayout>,
            RhsStageReader = PartialStageReader<RhsS<MP>, RL::TilingLayout>,
            AccStageReader = FillStageReader<AccS<MP>>,
            WriteCoords = Coords3d,
        >,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;
    type LhsStageLoader = SyncPartialStageLoader<MP::Lhs, Self::Config, LL>;
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
        let loop_step = stage_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = div_ceil(range, stage_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        let acc_reader = Self::AccStageLoader::reader(&acc_loader);
        SMM::load_accumulators(&acc_reader, acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let lhs_reader_a = Self::LhsStageLoader::reader(&lhs_loader, StageBuffer::A);
        let lhs_reader_b = Self::LhsStageLoader::reader(&lhs_loader, StageBuffer::B);
        let rhs_reader_a = Self::RhsStageLoader::reader(&rhs_loader, StageBuffer::A);
        let rhs_reader_b = Self::RhsStageLoader::reader(&rhs_loader, StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        load_first::<MP, SMM, Self::LhsStageLoader, Self::RhsStageLoader, Self::Config>(
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            StageBuffer::A,
            config,
        );

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_load_next::<
                MP,
                SMM,
                Self::LhsStageLoader,
                Self::RhsStageLoader,
                Self::Config,
            >(
                &lhs_reader_a,
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

            // We always advance by 2 * k because stage B shares the same global memory state as stage A,
            // but it is implicitly offset by one stage's worth (k elements) when reading.
            Self::LhsStageLoader::advance_view(&mut lhs_loader, loop_step);
            Self::RhsStageLoader::advance_view(&mut rhs_loader, loop_step);

            sync_cube();

            execute_current_and_load_next::<
                MP,
                SMM,
                Self::LhsStageLoader,
                Self::RhsStageLoader,
                Self::Config,
            >(
                &lhs_reader_b,
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

            sync_cube();
        }

        execute_current_and_load_next::<
            MP,
            SMM,
            Self::LhsStageLoader,
            Self::RhsStageLoader,
            Self::Config,
        >(
            &lhs_reader_a,
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

    fn init_lhs_stage_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let layout = SimpleGlobalLayout::new(&lhs, config.global_memory_config(MatmulIdent::Lhs));
        SyncPartialStageLoader::<MP::Lhs, Self::Config, LL>::new(
            lhs.view(layout),
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_stage_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        let layout = SimpleGlobalLayout::new(&rhs, config.global_memory_config(MatmulIdent::Rhs));
        SyncPartialStageLoader::<MP::Rhs, Self::Config, RL>::new(
            rhs.view(layout),
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_acc_stage_loader(
        acc: CubeOption<VirtualTensor<AccG<MP>>>,
        _m_offset: u32,
        _n_offset: u32,
        _nth_batch: u32,
        _batch_offset: u32,
        #[comptime] _config: Self::Config,
    ) -> Self::AccStageLoader {
        match acc {
            CubeOption::None => ZeroStageLoader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let layout = SimpleGlobalLayout::new(&out, config.global_memory_config(MatmulIdent::Out));
        SMM::init_writer(out.view_mut(layout), x_offset, y_offset, batch_offset)
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
