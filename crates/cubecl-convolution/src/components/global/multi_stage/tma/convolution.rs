use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
use cubecl_matmul::components::{
    LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        AccumulatorLoader, GlobalConfig as _, load::arrive_tma, single_stage::tma::SimpleTmaConfig,
    },
    layout::{Coords2d, VirtualTensorView},
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
            layout::NhwcOutGlobalLayout,
            load::{
                bias::BiasLoader,
                im2col_tma::{TmaIm2colLoader, TmaIm2colTiling},
                weight_tma::{TmaWeightLoader, TmaWeightTiling},
            },
        },
    },
    kernels::layered::selector::RuntimeArgs,
};

/// Performs convolution at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
///
/// Uses multiple stages to prefetch as much data as can fit into shared memory, reducing the impact
/// of memory latency. An example execution would look like this:
///
/// * Start loading stage 1, 2, 3, 4
/// * Wait for stage 1
/// * Execute with stage 1 data
/// * Refill stage 1 with the data for stage 5
/// * Wait for stage 2
/// * Execute with stage 2 data
/// * Refill stage 2 with the data for stage 6
///
/// Keep going until k is exhausted
pub struct MultiStageTmaConvolution<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> GlobalConvolution<MP> for MultiStageTmaConvolution<MP, SMM>
where
    SMM: StageMatmul<
            MP,
            LhsReader = FullStageToTileReader<LhsS<MP>, TmaIm2colTiling>,
            RhsReader = FullStageToTileReader<RhsS<MP>, TmaWeightTiling>,
            WriteCoords = Coords2d,
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
        // Arbitrarily using Lhs, they should be the same
        let num_stages = config.num_stages(MatmulIdent::Lhs);
        let stage_config = config.stage_config();
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;
        // Loop once for each full set of stages, then once for each stage in an inner loop,
        // so the stage index is comptime. This is needed to make `Sequence` work.
        let num_loops = (num_loops + num_stages - 1) / num_stages;

        let stage_elems_lhs = config.tiling_scheme().elements_in_stage_mk();
        let stage_elems_rhs = config.tiling_scheme().elements_in_stage_nk();

        Self::AccumulatorLoader::fill_stage::<Self::Config>(&mut acc_loader, config);

        sync_cube();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(&mut acc_loader, acc, stage_config);

        let mut lhs_barriers = Sequence::<Barrier<LhsS<MP>>>::new();
        let mut rhs_barriers = Sequence::<Barrier<RhsS<MP>>>::new();
        let (mut tile_lhs, mut tile_rhs) = SMM::init_tile_inputs(stage_config);

        let mut stage = comptime![0u32];

        // Create barriers and prefetch each stage
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..num_stages {
            let lhs_barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
            let rhs_barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

            Self::LhsLoader::fill_stage(&mut lhs_loader, &lhs_barrier, stage, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &rhs_barrier, stage, stage_config);

            arrive_tma::<LhsS<MP>>(&lhs_barrier, stage_elems_lhs);
            arrive_tma::<RhsS<MP>>(&rhs_barrier, stage_elems_rhs);

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            lhs_barriers.push(lhs_barrier);
            rhs_barriers.push(rhs_barrier);

            comptime![stage += 1];
        }

        for k in 0..num_loops {
            let k = k * num_stages;

            let mut stage = comptime![0u32];

            // Loop through all stages
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..num_stages {
                let k = k + stage;
                let next_k = k + num_stages;

                // Bounds check for k stage, for when `k_stages % num_stages != 0`
                if k < k_range.1 {
                    let lhs_barrier = lhs_barriers.index(stage);
                    let rhs_barrier = rhs_barriers.index(stage);

                    let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader, stage);
                    let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader, stage);

                    // Wait for load and execute matmul on this stage
                    lhs_barrier.wait();
                    rhs_barrier.wait();
                    SMM::execute(
                        lhs_stage_reader,
                        rhs_stage_reader,
                        &mut tile_lhs,
                        &mut tile_rhs,
                        acc,
                        config.stage_config(),
                    );
                    lhs_barrier.arrive();
                    rhs_barrier.arrive();

                    // Check if there's any stages left to load in the k dimension
                    if next_k < k_range.1 {
                        lhs_barrier.wait();
                        rhs_barrier.wait();

                        // Refill stage and advance view
                        Self::LhsLoader::fill_stage(&mut lhs_loader, lhs_barrier, stage, config);
                        Self::RhsLoader::fill_stage(
                            &mut rhs_loader,
                            rhs_barrier,
                            stage,
                            stage_config,
                        );

                        arrive_tma::<LhsS<MP>>(lhs_barrier, stage_elems_lhs);
                        arrive_tma::<RhsS<MP>>(rhs_barrier, stage_elems_rhs);

                        Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
                        Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
                    }
                }

                comptime![stage += 1];
            }
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
        Self::LhsLoader::new(
            lhs,
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(MatmulIdent::Lhs),
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
        Self::RhsLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(MatmulIdent::Rhs),
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
        let layout = NhwcOutGlobalLayout::new(
            &out,
            runtime_args.size_m,
            runtime_args.size_n,
            runtime_args.out_shape.clone(),
            config.global_memory_config(MatmulIdent::Out),
        );
        let out = VirtualTensorView::new(out, layout.into_virtual());
        SMM::init_writer(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }
}
