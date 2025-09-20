use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalConfig as _,
        load::{SyncFullStageLoader, sync_full_cyclic},
        single_stage::simple::SimpleConfig,
    },
    stage::{FullStageReader, RowMajorTilingOrder, StageMatmul},
};
use cubecl_std::{
    CubeOption,
    tensor::{layout::Coords2d, r#virtual::VirtualTensor},
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig,
        global::{
            ConvTilingLayout, GlobalConvolution,
            layout::{Im2colLayout, NhwcLayout, OutLayout, WeightLayout},
            load::bias::{BiasStageLoader, BiasStageReader},
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
            LhsStageReader = FullStageReader<LhsS<MP>, ConvTilingLayout>,
            RhsStageReader = FullStageReader<RhsS<MP>, ConvTilingLayout>,
            AccStageReader = BiasStageReader<AccS<MP>>,
            WriteCoords = Coords2d,
        >,
{
    type LhsStageLoader = SyncFullStageLoader<
        MP::Lhs,
        Self::Config,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
    type Config = ConvolutionConfig<SimpleConfig<SMM::Config>>;
    type RhsStageLoader = SyncFullStageLoader<
        MP::Rhs,
        Self::Config,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
    type AccStageLoader = BiasStageLoader<MP::Acc>;

    type StageUnloader = SMM::StageUnloader;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_loader: Self::LhsStageLoader,
        mut rhs_loader: Self::RhsStageLoader,
        mut acc_loader: Self::AccStageLoader,
        mut out_writer: Self::StageUnloader,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        acc_loader.load_stage::<Self::Config>(config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::load_accumulators(&acc_loader.reader(), acc, config.stage_config());

        for _ in 0..num_loops {
            sync_cube();

            lhs_loader.load_stage(config);
            rhs_loader.load_stage(config);

            sync_cube();

            SMM::execute(
                &lhs_loader.reader(),
                &rhs_loader.reader(),
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            lhs_loader.advance_view();
            rhs_loader.advance_view();
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
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let check_spatial = comptime![config.check_spatial_bounds()];
        let layout_global = NhwcLayout::new(lhs, comptime![config.dimensionality()], check_spatial);
        let layout_im2col = Im2colLayout::new(runtime_args, config);
        let lhs = lhs.view(layout_global).view(layout_im2col);
        Self::LhsStageLoader::new(
            lhs.slice_unchecked(offset, slice_size),
            config.k_step,
            MatmulIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        let layout_global = NhwcLayout::new(rhs, comptime![config.dimensionality()], false);
        let layout_weight = WeightLayout::new(&rhs, runtime_args, config);
        let rhs = rhs.view(layout_global).view(layout_weight);
        Self::RhsStageLoader::new(
            rhs.slice_unchecked(offset, slice_size),
            config.k_step,
            MatmulIdent::Rhs,
            config,
        )
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        slice_size: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccStageLoader {
        Self::AccStageLoader::new::<Self::Config>(bias, n_offset, slice_size, config)
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let global_conf = config.global_memory_config(MatmulIdent::Out);
        let layout_global = NhwcLayout::new(out, comptime![config.dimensionality()], false);
        let layout_out = OutLayout::new(runtime_args, global_conf);
        let out = out.view_mut(layout_global).view_mut(layout_out);
        SMM::init_writer(out.slice_mut_unchecked(offset, slice_size), global_conf)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
