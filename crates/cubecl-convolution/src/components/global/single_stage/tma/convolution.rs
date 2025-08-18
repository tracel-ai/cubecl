use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
use cubecl_matmul::components::{
    LhsG, LhsS, MatmulPrecision, RhsG, RhsS,
    global::{
        AccumulatorLoader, GlobalConfig as _, load::arrive_tma, single_stage::tma::SimpleTmaConfig,
    },
    stage::{FullStageToTileReader, StageMatmul},
};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use crate::{
    components::{
        ConvolutionConfig,
        global::{
            GlobalConvolution,
            load::{
                bias::BiasLoader,
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
            LhsReader = FullStageToTileReader<LhsS<MP>, TmaIm2colTiling>,
            RhsReader = FullStageToTileReader<RhsS<MP>, TmaWeightTiling>,
        >,
{
    type LhsLoader = TmaIm2colLoader<MP::Lhs, Self::Config>;
    type Config = ConvolutionConfig<SimpleTmaConfig<SMM::Config>>;
    type RhsLoader = TmaWeightLoader<MP::Rhs, SMM::Config>;
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

        let stage_elems_lhs = config.tiling_scheme().elements_in_stage_mk();
        let stage_elems_rhs = config.tiling_scheme().elements_in_stage_nk();

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());

        sync_cube();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.stage_config(),
        );

        let lhs_barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
        let rhs_barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsLoader::fill_stage(&mut lhs_loader, &lhs_barrier, 0u32, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &rhs_barrier, 0u32, config.stage_config());

            arrive_tma::<LhsS<MP>>(&lhs_barrier, stage_elems_lhs);
            arrive_tma::<RhsS<MP>>(&rhs_barrier, stage_elems_rhs);

            lhs_barrier.wait();
            rhs_barrier.wait();

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader, 0u32);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader, 0u32);

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

        sync_cube();

        SMM::write_results::<Self::Config>(acc, &mut out_writer, config.stage_config(), config);
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(lhs, x_offset, y_offset, runtime_args, 1u32, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            1u32,
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
    ) -> Self::Writer {
        SMM::init_writer(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
