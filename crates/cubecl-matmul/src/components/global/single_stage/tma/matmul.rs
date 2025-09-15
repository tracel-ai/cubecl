use crate::components::RhsS;
use crate::components::global::GlobalMatmul;
use crate::components::global::load::TmaLoader;
use crate::components::global::load::TmaReader;
use crate::components::global::load::arrive_tma;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::stage::StageMatmul;
use crate::components::{AccG, RhsG};
use crate::components::{AccS, MatmulIdent};
use crate::components::{LhsG, global::memory::SimpleGlobalLayout};
use crate::components::{LhsS, stage::FillReader};
use crate::components::{MatmulPrecision, global::load::ZeroLoader};
use barrier::Barrier;
use cubecl_core::prelude::{barrier::BarrierLevel, *};
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords3d};
use std::marker::PhantomData;

use crate::components::global::GlobalConfig;

/// Performs matrix multiplication at the global level
/// Similar to simple matmul but using tma loading
pub struct SimpleTmaMatmul<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _phantom: PhantomData<(MP, SMM)>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> GlobalMatmul<MP> for SimpleTmaMatmul<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsReader = TmaReader<MP::Lhs>,
            RhsReader = TmaReader<MP::Rhs>,
            AccReader = FillReader<AccS<MP>>,
            WriteCoords = Coords3d,
        >,
{
    type Config = SimpleTmaConfig<SMM::Config>;
    type LhsLoader = TmaLoader<MP::Lhs, Self::Config>;
    type RhsLoader = TmaLoader<MP::Rhs, Self::Config>;
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

        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let num_bytes_lhs =
            comptime!(config.tiling_scheme().elements_in_stage_mk() * lhs_elem_size);
        let num_bytes_rhs =
            comptime!(config.tiling_scheme().elements_in_stage_nk() * rhs_elem_size);
        let num_bytes_stages = num_bytes_lhs + num_bytes_rhs;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        SMM::fill_accumulator(&acc_loader.reader(), acc, config.stage_config());

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            // Start loading
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier, config);

            arrive_tma(&barrier, num_bytes_stages);

            barrier.wait();

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

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
        nth_batch: u32,
        _batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        _batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
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
