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
    tensor::{layout::Coords2d, r#virtual::VirtualTensor},
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
            LhsStageReader = FullStageReader<LhsS<MP>, TmaIm2colTiling>,
            RhsStageReader = FullStageReader<RhsS<MP>, TmaWeightTiling>,
            AccStageReader = BiasStageReader<AccS<MP>>,
            WriteCoords = Coords2d,
        >,
{
    type Config = ConvolutionConfig<SimpleTmaConfig<SMM::Config>>;

    type LhsStageLoader = TmaIm2colLoader<MP::Lhs, Self::Config>;
    type RhsStageLoader = TmaWeightLoader<MP::Rhs, SMM::Config>;
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
        // Arbitrarily using Lhs, they should be the same
        let num_stages = config.num_stages(MatmulIdent::Lhs);
        let stage_config = config.stage_config();
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);
        // Loop once for each full set of stages, then once for each stage in an inner loop,
        // so the stage index is comptime. This is needed to make `Sequence` work.
        let num_loops = num_loops.div_ceil(num_stages);

        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let stage_bytes_lhs =
            comptime![config.tiling_scheme().elements_in_stage_mk() * lhs_elem_size];
        let stage_bytes_rhs =
            comptime![config.tiling_scheme().elements_in_stage_nk() * rhs_elem_size];
        let stages_bytes = stage_bytes_lhs + stage_bytes_rhs;

        Self::AccStageLoader::load_stage::<Self::Config>(&mut acc_loader, config);

        sync_cube();

        SMM::load_accumulators(&acc_loader.reader(), acc, stage_config);

        let mut barriers = Sequence::<Barrier>::new();
        let (mut tile_lhs, mut tile_rhs) = SMM::init_tile_inputs(stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let mut stage = comptime![0u32];

        // Create barriers and prefetch each stage
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..num_stages {
            let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

            Self::LhsStageLoader::fill_stage(&mut lhs_loader, &barrier, stage, config);
            Self::RhsStageLoader::fill_stage(&mut rhs_loader, &barrier, stage, stage_config);

            arrive_tma(&barrier, stages_bytes);

            Self::LhsStageLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsStageLoader::advance_view(&mut rhs_loader, k_step);

            barriers.push(barrier);

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
                    let barrier = barriers.index(stage);

                    let lhs_stage_reader = &Self::LhsStageLoader::reader(&lhs_loader, stage);
                    let rhs_stage_reader = &Self::RhsStageLoader::reader(&rhs_loader, stage);

                    // Wait for load and execute matmul on this stage
                    barrier.wait();
                    SMM::execute(
                        lhs_stage_reader,
                        rhs_stage_reader,
                        &mut tile_lhs,
                        &mut tile_rhs,
                        acc,
                        config.stage_config(),
                        &partition_scheduler,
                    );
                    barrier.arrive();

                    // Check if there's any stages left to load in the k dimension
                    if next_k < k_range.1 {
                        barrier.wait();

                        // Refill stage and advance view
                        Self::LhsStageLoader::fill_stage(&mut lhs_loader, barrier, stage, config);
                        Self::RhsStageLoader::fill_stage(
                            &mut rhs_loader,
                            barrier,
                            stage,
                            stage_config,
                        );

                        arrive_tma(barrier, stages_bytes);

                        Self::LhsStageLoader::advance_view(&mut lhs_loader, k_step);
                        Self::RhsStageLoader::advance_view(&mut rhs_loader, k_step);
                    }
                }

                comptime![stage += 1];
            }
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
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader {
        let (x_offset, y_offset) = offset;
        Self::LhsStageLoader::new(
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
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader {
        let (x_offset, y_offset) = offset;
        Self::RhsStageLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(MatmulIdent::Rhs),
            config,
        )
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        _slice_size: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccStageLoader {
        Self::AccStageLoader::new::<Self::Config>(bias, n_offset, config)
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::StageUnloader {
        let layout_global = NhwcLayout::new(out, comptime![config.dimensionality()], false);
        let layout_out =
            OutLayout::new(runtime_args, config.global_memory_config(MatmulIdent::Out));
        let out = out.view_mut(layout_global).view_mut(layout_out);
        SMM::init_writer(out.slice_mut_unchecked(offset, slice_size))
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
