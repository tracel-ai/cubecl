use crate::components::RhsS;
use crate::components::global::GlobalMatmul;
use crate::components::global::load::TmaStageLoader;
use crate::components::global::load::TmaStageReader;
use crate::components::global::load::arrive_tma;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::stage::StageMatmul;
use crate::components::{AccG, RhsG};
use crate::components::{AccS, MatmulIdent};
use crate::components::{LhsG, global::memory::SimpleGlobalLayout};
use crate::components::{LhsS, stage::FillStageReader};
use crate::components::{MatmulPrecision, global::load::ZeroStageLoader};
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
            LhsStageReader = TmaStageReader<MP::Lhs>,
            RhsStageReader = TmaStageReader<MP::Rhs>,
            AccStageReader = FillStageReader<AccS<MP>>,
            WriteCoords = Coords3d,
        >,
{
    type Config = SimpleTmaConfig<SMM::Config>;
    type LhsStageLoader = TmaStageLoader<MP::Lhs, Self::Config>;
    type RhsStageLoader = TmaStageLoader<MP::Rhs, Self::Config>;
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

        SMM::load_accumulators(&acc_loader.reader(), acc, config.stage_config());

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            // Start loading
            Self::LhsStageLoader::load_stage(&mut lhs_loader, &barrier, config);
            Self::RhsStageLoader::load_stage(&mut rhs_loader, &barrier, config);

            arrive_tma(&barrier, num_bytes_stages);

            barrier.wait();

            let lhs_stage_reader = &Self::LhsStageLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsStageLoader::reader(&rhs_loader);

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
        offset: Coords3d,
        _slice_size: Coords3d,
        nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let (_, x_offset, y_offset) = offset;
        Self::LhsStageLoader::new(
            lhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_stage_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        offset: Coords3d,
        _slice_size: Coords3d,
        nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        let (_, x_offset, y_offset) = offset;
        Self::RhsStageLoader::new(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_acc_stage_loader(
        acc: CubeOption<VirtualTensor<AccG<MP>>>,
        _offset: Coords3d,
        _slice_size: Coords3d,
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
        offset: Coords3d,
        slice_size: Coords3d,
        _nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let layout = SimpleGlobalLayout::new(&out, config.global_memory_config(MatmulIdent::Out));
        SMM::init_writer(out.view_mut(layout).slice_mut_unchecked(offset, slice_size))
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
