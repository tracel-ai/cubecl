use std::marker::PhantomData;

use crate::components::MatmulPrecision;
use crate::components::RhsG;
use crate::components::RhsS;
use crate::components::global::GlobalConfig;
use crate::components::global::GlobalMatmul;
use crate::components::global::load::AsyncFullLoader;
use crate::components::global::load::AsyncFullLoadingStrategy;
use crate::components::global::memory::SimpleGlobalLayout;
use crate::components::global::single_stage::barrier::SimpleBarrierConfig;
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::StageMatmul;
use crate::components::{AccG, AccS, LhsS};
use crate::components::{LhsG, global::load::ZeroLoader};
use crate::components::{MatmulIdent, stage::FillReader};
use barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords3d};

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
            LhsReader = FullStageToTileReader<LhsS<MP>, LL::TilingLayout>,
            RhsReader = FullStageToTileReader<RhsS<MP>, RL::TilingLayout>,
            AccReader = FillReader<AccS<MP>>,
            WriteCoords = Coords3d,
        >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Config = SimpleBarrierConfig<SMM::Config>;
    type LhsLoader = AsyncFullLoader<MP::Lhs, Barrier, SMM::Config, LL, Self::Config>;
    type RhsLoader = AsyncFullLoader<MP::Rhs, Barrier, SMM::Config, RL, Self::Config>;
    type AccLoader = ZeroLoader<MP::Acc>;
    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        acc_loader: Self::AccLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let acc_reader = acc_loader.reader();
        SMM::fill_accumulator(&acc_reader, acc, config.stage_config());

        let barrier_level = LL::barrier_level();
        let lhs_barrier = Barrier::new(barrier_level);
        let rhs_barrier = Barrier::new(barrier_level);

        for loop_iter in 0..num_loops {
            sync_cube();

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, config);
                    sync_cube();
                }
            }

            // Start loading
            Self::LhsLoader::fill_stage(&mut lhs_loader, &lhs_barrier, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &rhs_barrier, config);

            let lhs_stage_reader = &lhs_loader.reader();
            let rhs_stage_reader = &rhs_loader.reader();

            lhs_barrier.arrive_and_wait();
            rhs_barrier.arrive_and_wait();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::write_results::<Self::Config>(
            acc,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        let layout = SimpleGlobalLayout::new(&lhs, config.global_memory_config(MatmulIdent::Lhs));
        Self::LhsLoader::new(
            lhs.view(layout),
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        let layout = SimpleGlobalLayout::new(&rhs, config.global_memory_config(MatmulIdent::Rhs));
        Self::RhsLoader::new(
            rhs.view(layout),
            x_offset,
            y_offset,
            batch_offset,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_acc_loader(
        acc: CubeOption<VirtualTensor<AccG<MP>>>,
        _m_offset: u32,
        _n_offset: u32,
        _nth_batch: u32,
        _batch_offset: u32,
        #[comptime] _config: Self::Config,
    ) -> Self::AccLoader {
        match acc {
            CubeOption::None => ZeroLoader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let layout = SimpleGlobalLayout::new(&out, config.global_memory_config(MatmulIdent::Out));
        SMM::init_writer(out.view_mut(layout), x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
