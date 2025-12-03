use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalConfig as _, GlobalWriter, PartitionedStage, PlaneWriter, SharedGlobalMatmulConfig,
    },
    stage::{StageConfig, StageMatmul, StridedStageMemory},
};
use cubecl_std::{
    CubeOption,
    tensor::{View, layout::Coords2d},
};

use crate::components::{
    ConvolutionConfig,
    global::{
        GlobalConvolution,
        args::RuntimeArgs,
        read::{
            bias::{BiasGlobalReader, BiasStage},
            im2col_tma::{TmaIm2colGlobalReader, TmaIm2colTiling},
            weight_tma::{TmaWeightGlobalReader, TmaWeightTiling},
        },
    },
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
            LhsStage = StridedStageMemory<LhsS<MP>, TmaIm2colTiling>,
            RhsStage = StridedStageMemory<RhsS<MP>, TmaWeightTiling>,
            AccStage = BiasStage<AccS<MP>>,
            OutStage = PartitionedStage<AccS<MP>>,
        >,
{
    type Config = ConvolutionConfig<SharedGlobalMatmulConfig<SMM::Config>>;

    type LhsGlobalReader = TmaIm2colGlobalReader<MP::Lhs>;
    type RhsGlobalReader = TmaWeightGlobalReader<MP::Rhs>;
    type AccGlobalReader = BiasGlobalReader<MP::Acc>;
    type GlobalWriter = PlaneWriter<MP::Acc>;

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
        let num_stages = config.num_stages;
        let stage_config = config.stage_config();
        let k_step = config.stage_config.elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);
        // Loop once for each full set of stages, then once for each stage in an inner loop,
        // so the stage index is comptime. This is needed to make `Sequence` work.
        let num_loops = num_loops.div_ceil(num_stages);

        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let stage_bytes_lhs =
            comptime![stage_config.elements_in_stage_m() * k_step * lhs_elem_size];
        let stage_bytes_rhs =
            comptime![stage_config.elements_in_stage_n() * k_step * rhs_elem_size];
        let stages_bytes = stage_bytes_lhs + stage_bytes_rhs;

        acc_reader.load_stage::<SharedGlobalMatmulConfig<SMM::Config>>(config.matmul);

        sync_cube();

        SMM::load_accumulators(&acc_reader.stage(), acc, stage_config);

        let mut barriers_full = Sequence::<Shared<Barrier>>::new();
        let mut barriers_empty = Sequence::<Shared<Barrier>>::new();
        let (mut tile_lhs, mut tile_rhs) = SMM::init_tile_inputs(stage_config);
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        // Create barriers and prefetch each stage
        #[unroll]
        for stage in 0..num_stages {
            let barrier_full = Barrier::shared(1, UNIT_POS == 0u32);
            let barrier_empty = Barrier::shared(CUBE_DIM, UNIT_POS == 0u32);

            lhs_reader.fill_stage(&barrier_full, stage);
            rhs_reader.fill_stage(&barrier_full, stage);

            if UNIT_POS == 0 {
                barrier_full.arrive_and_expect_tx(1, stages_bytes);
            }

            lhs_reader.advance_view(k_step);
            rhs_reader.advance_view();

            barriers_full.push(barrier_full);
            barriers_empty.push(barrier_empty);
        }

        let mut phase = 0;

        for k in 0..num_loops {
            let k = k * num_stages;

            // Loop through all stages
            #[unroll]
            for stage in 0..num_stages {
                let k = k + stage;
                let next_k = k + num_stages;

                // Bounds check for k stage, for when `k_stages % num_stages != 0`
                if k < k_range.1 {
                    let barrier_full = barriers_full.index(stage);
                    let barrier_empty = barriers_empty.index(stage);

                    // Wait for load and execute matmul on this stage
                    barrier_full.wait_parity(phase);
                    SMM::execute(
                        &lhs_reader.stage(stage),
                        &rhs_reader.stage(stage),
                        &mut tile_lhs,
                        &mut tile_rhs,
                        acc,
                        config.stage_config(),
                        &partition_scheduler,
                    );
                    barrier_empty.arrive_and_wait();

                    // Check if there's any stages left to load in the k dimension
                    if next_k < k_range.1 {
                        // Refill stage and advance view
                        lhs_reader.fill_stage(barrier_full, stage);
                        rhs_reader.fill_stage(barrier_full, stage);

                        if UNIT_POS == 0 {
                            barrier_full.arrive_and_expect_tx(1, stages_bytes);
                        }

                        lhs_reader.advance_view(k_step);
                        rhs_reader.advance_view();
                    }
                }
            }

            phase ^= 1;
        }

        sync_cube();

        let mut out_stage = Self::GlobalWriter::stage(&out_writer);

        SMM::write_results::<Self::GlobalWriter>(
            acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        let (x_offset, y_offset) = offset;
        Self::LhsGlobalReader::new(
            lhs.as_tensor_map().unwrap(),
            x_offset,
            y_offset,
            runtime_args,
            config.num_stages,
            config.convolution_params,
            config.lhs_reader_config.smem_config,
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        _runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(
            rhs,
            config.stage_config.elements_in_stage_k(),
            config.num_stages,
            config.rhs_reader_config.smem_config,
        )
    }

    fn init_bias_global_reader(
        bias: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        Self::AccGlobalReader::new(bias, config.writer_config.smem_config)
    }

    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        Self::GlobalWriter::new(out, config.writer_config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
