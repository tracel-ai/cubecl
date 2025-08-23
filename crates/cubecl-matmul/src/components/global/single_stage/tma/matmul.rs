use crate::components::LhsS;
use crate::components::MatmulIdent;
use crate::components::MatmulPrecision;
use crate::components::RhsG;
use crate::components::RhsS;
use crate::components::global::GlobalMatmul;
use crate::components::global::ZeroAccumulatorLoader;
use crate::components::global::load::TmaLoader;
use crate::components::global::load::TmaReader;
use crate::components::global::load::arrive_tma;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::stage::StageMatmul;
use crate::components::{LhsG, global::memory::SimpleGlobalLayout};
use barrier::Barrier;
use cubecl_core::prelude::{barrier::BarrierLevel, *};
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::tensor::{layout::Coords3d, r#virtual::ReadWrite};
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
            WriteCoords = Coords3d,
        >,
{
    type Config = SimpleTmaConfig<SMM::Config>;
    type LhsLoader = TmaLoader<MP::Lhs, Self::Config>;
    type RhsLoader = TmaLoader<MP::Rhs, Self::Config>;
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
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;
        let num_elems_lhs = config.tiling_scheme().elements_in_stage_mk();
        let num_elems_rhs = config.tiling_scheme().elements_in_stage_nk();

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        SMM::zero_accumulator(acc, config.stage_config());

        let barrier_lhs = Barrier::<LhsS<MP>>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
        let barrier_rhs = Barrier::<RhsS<MP>>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            // Start loading
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier_lhs, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier_rhs, config);

            arrive_tma::<LhsS<MP>>(&barrier_lhs, num_elems_lhs);
            arrive_tma::<RhsS<MP>>(&barrier_rhs, num_elems_rhs);

            barrier_lhs.wait();
            barrier_rhs.wait();

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::write_results::<Self::Config>(acc, &mut out_writer, config.stage_config(), config);
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

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let layout = SimpleGlobalLayout::new(&out, config.global_memory_config(MatmulIdent::Out));
        SMM::init_writer(
            out.view_mut(layout.virt()),
            x_offset,
            y_offset,
            batch_offset,
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
