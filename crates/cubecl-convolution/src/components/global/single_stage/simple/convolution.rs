use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        AccumulatorLoader, GlobalConfig as _,
        load::{SyncFullLoader, sync_full_cyclic},
        single_stage::simple::SimpleConfig,
    },
    stage::{FullStageToTileReader, RowMajorTilingOrder, StageMatmul},
};
use cubecl_std::{
    CubeOption,
    tensor::{layout::Coords3d, r#virtual::VirtualTensor},
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig,
        global::{
            ConvTilingLayout, GlobalConvolution,
            layout::{Im2colGlobalLayout, NhwcLayout, OutLayout, WeightLayout},
            load::bias::BiasLoader,
        },
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimpleConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> GlobalConvolution<MP> for SimpleConvolution<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsReader = FullStageToTileReader<LhsS<MP>, ConvTilingLayout>,
            RhsReader = FullStageToTileReader<RhsS<MP>, ConvTilingLayout>,
            WriteCoords = Coords3d,
        >,
{
    type LhsLoader = SyncFullLoader<
        MP::Lhs,
        Self::Config,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
    type Config = ConvolutionConfig<SimpleConfig<SMM::Config>>;
    type RhsLoader = SyncFullLoader<
        MP::Rhs,
        Self::Config,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
    type AccumulatorLoader = BiasLoader<MP>;

    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.stage_config(),
        );

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

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

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        sync_cube();

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
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        let check_spatial = comptime![config.check_spatial_bounds()];
        let layout_global =
            NhwcLayout::new(lhs, comptime![config.dimensionality()], check_spatial).virt();
        let layout_im2col = Im2colGlobalLayout::new(runtime_args, config).virt();
        Self::LhsLoader::new(
            lhs.view(layout_global).view(layout_im2col),
            x_offset,
            y_offset,
            0,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        let layout_global = NhwcLayout::new(rhs, comptime![config.dimensionality()], false).virt();
        let layout_weight = WeightLayout::new(&rhs, runtime_args, config).virt();
        Self::RhsLoader::new(
            rhs.view(layout_global).view(layout_weight),
            x_offset,
            y_offset,
            0,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<MP::EO>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccumulatorLoader {
        Self::AccumulatorLoader::new::<Self::Config>(bias, n_offset, config)
    }

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let layout_global = NhwcLayout::new(out, comptime![config.dimensionality()], false).virt();
        let layout_out =
            OutLayout::new(runtime_args, config.global_memory_config(MatmulIdent::Out)).virt();
        SMM::init_writer(
            out.view_mut(layout_global).view_mut(layout_out),
            x_offset,
            y_offset,
            0,
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
