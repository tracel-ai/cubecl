use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{GlobalConfig as _, read::arrive_tma, single_stage::tma::SimpleTmaConfig},
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
            read::{
                bias::{BiasGlobalReader, BiasStageReader},
                im2col_tma::{TmaIm2colGlobalReader, TmaIm2colTiling},
                weight_tma::{TmaWeightGlobalReader, TmaWeightTiling},
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

    type LhsGlobalReader = TmaIm2colGlobalReader<MP::Lhs, Self::Config>;
    type RhsGlobalReader = TmaWeightGlobalReader<MP::Rhs, SMM::Config>;
    type AccGlobalReader = BiasGlobalReader<MP::Acc>;

    type GlobalWriter = SMM::GlobalWriter;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        mut acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
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

        acc_reader.load_stage::<Self::Config>(config);

        sync_cube();

        SMM::load_accumulators(&acc_reader.stage_reader(), acc, stage_config);

        let mut barriers = Sequence::<Barrier>::new();
        let (mut tile_lhs, mut tile_rhs) = SMM::init_tile_inputs(stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        let mut stage = comptime![0u32];

        // Create barriers and prefetch each stage
        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..num_stages {
            let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

            lhs_reader.fill_stage(&barrier, stage, config);
            rhs_reader.fill_stage(&barrier, stage, stage_config);

            arrive_tma(&barrier, stages_bytes);

            lhs_reader.advance_view(k_step);
            rhs_reader.advance_view(k_step);

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

                    // Wait for load and execute matmul on this stage
                    barrier.wait();
                    SMM::execute(
                        &lhs_reader.stage_reader(stage),
                        &rhs_reader.stage_reader(stage),
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
                        lhs_reader.fill_stage(barrier, stage, config);
                        rhs_reader.fill_stage(barrier, stage, stage_config);

                        arrive_tma(barrier, stages_bytes);

                        lhs_reader.advance_view(k_step);
                        rhs_reader.advance_view(k_step);
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

    fn init_lhs_global_reader(
        lhs: VirtualTensor<LhsG<MP>>,
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        let (x_offset, y_offset) = offset;
        Self::LhsGlobalReader::new(
            lhs,
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(MatmulIdent::Lhs),
            config,
        )
    }

    fn init_rhs_global_reader(
        rhs: VirtualTensor<RhsG<MP>>,
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        let (x_offset, y_offset) = offset;
        Self::RhsGlobalReader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages(MatmulIdent::Rhs),
            config,
        )
    }

    fn init_bias_global_reader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        slice_size: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        Self::AccGlobalReader::new::<Self::Config>(bias, n_offset, slice_size, config)
    }

    fn init_global_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
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
