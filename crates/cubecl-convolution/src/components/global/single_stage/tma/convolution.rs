use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{GlobalConfig as _, load::arrive_tma, single_stage::tma::SimpleTmaConfig},
    stage::{FullStageReader, StageMatmul},
};
use cubecl_std::{
    CubeOption,
    tensor::{layout::Coords3d, r#virtual::VirtualTensor},
};

use crate::{
    components::{
        ConvGemmConfig, ConvolutionConfig,
        global::{
            GlobalConvolution,
            layout::{NhwcLayout, OutLayout},
            load::{
                bias::{BiasStageLoader, BiasStageReader},
                im2col_tma::{TmaIm2colLoader, TmaIm2colTiling},
                weight_tma::{TmaWeightLoader, TmaWeightTiling},
            },
        },
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimpleTmaConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> GlobalConvolution<MP> for SimpleTmaConvolution<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsStageReader = FullStageReader<LhsS<MP>, TmaIm2colTiling>,
            RhsStageReader = FullStageReader<RhsS<MP>, TmaWeightTiling>,
            AccStageReader = BiasStageReader<AccS<MP>>,
            WriteCoords = Coords3d,
        >,
{
    type Config = ConvolutionConfig<SimpleTmaConfig<SMM::Config>>;

    type LhsStageLoader = TmaIm2colLoader<MP::Lhs, Self::Config>;
    type RhsStageLoader = TmaWeightLoader<MP::Rhs, SMM::Config>;
    type AccStageLoader = BiasStageLoader<MP::Acc>;

    type StageWriter = SMM::StageWriter;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_loader: Self::LhsStageLoader,
        mut rhs_loader: Self::RhsStageLoader,
        mut acc_loader: Self::AccStageLoader,
        mut out_writer: Self::StageWriter,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let stage_bytes_lhs =
            comptime!(config.tiling_scheme().elements_in_stage_mk() * lhs_elem_size);
        let stage_bytes_rhs =
            comptime!(config.tiling_scheme().elements_in_stage_nk() * rhs_elem_size);
        let stages_bytes = stage_bytes_lhs + stage_bytes_rhs;

        Self::AccStageLoader::load_stage::<Self::Config>(&mut acc_loader, config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::load_accumulators(&acc_loader.reader(), acc, config.stage_config());

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsStageLoader::fill_stage(&mut lhs_loader, &barrier, 0u32, config);
            Self::RhsStageLoader::fill_stage(
                &mut rhs_loader,
                &barrier,
                0u32,
                config.stage_config(),
            );

            arrive_tma(&barrier, stages_bytes);

            barrier.wait();

            let lhs_stage_reader = &Self::LhsStageLoader::reader(&lhs_loader, 0u32);
            let rhs_stage_reader = &Self::RhsStageLoader::reader(&rhs_loader, 0u32);

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
    ) -> Self::LhsStageLoader {
        Self::LhsStageLoader::new(lhs, x_offset, y_offset, runtime_args, 1u32, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        Self::RhsStageLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            1u32,
            config,
        )
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccStageLoader {
        Self::AccStageLoader::new::<Self::Config>(bias, n_offset, config)
    }

    fn init_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::StageWriter {
        let layout_global = NhwcLayout::new(out, comptime![config.dimensionality()], false);
        let layout_out =
            OutLayout::new(runtime_args, config.global_memory_config(MatmulIdent::Out));
        SMM::init_writer(
            out.view_mut(layout_global).view_mut(layout_out),
            x_offset,
            y_offset,
            0,
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
