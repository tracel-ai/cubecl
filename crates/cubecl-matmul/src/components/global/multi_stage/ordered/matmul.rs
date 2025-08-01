use crate::components::global::load::{
    StageBuffer, SyncFullLoader, SyncFullLoadingStrategy, SyncPartialLoader,
    SyncPartialLoadingStrategy,
};
use crate::components::global::multi_stage::double_buffer_execution::{
    execute_current_and_load_next, execute_last_and_write_results, load_first,
};
use crate::components::global::multi_stage::ordered::LL;
use crate::components::global::{self, GlobalConfig, ZeroAccumulatorLoader};
use crate::components::global::{Quantization, Specializer};
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::PartialStageToTileReader;
use crate::components::{MatmulIdent, MatmulPrecision, stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};
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
            LhsReader = FullStageToTileReader<
                MP::ES,
                <LL as SyncFullLoadingStrategy>::TilingLayout,
            >,
            RhsReader = PartialStageToTileReader<MP::ES, RL::TilingLayout>,
        >,
    RL: SyncPartialLoadingStrategy,
{
    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = SyncFullLoader<MP, Self::Config, LL>;
    type RhsLoader = SyncPartialLoader<MP, Self::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
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

        SMM::zero_accumulator(acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());

        let lhs_reader = Self::LhsLoader::reader(&lhs_loader);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader, StageBuffer::A);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader, StageBuffer::B);

        let specializer = Specializer::new::<Self::Config>(config);

        load_first::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            StageBuffer::A,
            config,
        );

        Self::LhsLoader::advance_view(&mut lhs_loader, stage_step);

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                StageBuffer::B,
                config,
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, stage_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, loop_step);

            sync_cube();

            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                StageBuffer::A,
                config,
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, stage_step);

            sync_cube();
        }

        execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
            &lhs_reader,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
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
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        SyncFullLoader::<MP, Self::Config, LL>::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        SyncPartialLoader::<MP, Self::Config, RL>::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SMM::init_writer(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
