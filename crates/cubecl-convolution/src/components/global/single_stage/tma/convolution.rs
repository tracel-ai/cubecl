use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalConfig as _, GlobalWriter, PartitionedStage, PlaneWriter, SharedGlobalMatmulConfig,
        read::async_tma::arrive_tma,
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
        let k_step = config.stage_config.elements_in_stage_k();
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let stage_bytes_lhs =
            comptime!(config.stage_config.elements_in_stage_m() * k_step * lhs_elem_size);
        let stage_bytes_rhs =
            comptime!(config.stage_config.elements_in_stage_n() * k_step * rhs_elem_size);
        let stages_bytes = stage_bytes_lhs + stage_bytes_rhs;

        acc_reader.load_stage::<SharedGlobalMatmulConfig<SMM::Config>>(config.matmul);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0u32);
        sync_async_proxy_shared();

        for _ in 0..num_loops {
            sync_cube();

            lhs_reader.fill_stage(&barrier, 0u32);
            rhs_reader.fill_stage(&barrier, 0u32);

            let token = arrive_tma(&barrier, stages_bytes);

            barrier.wait(token);

            SMM::execute(
                &lhs_reader.stage(0u32),
                &rhs_reader.stage(0u32),
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            lhs_reader.advance_view(k_step);
            rhs_reader.advance_view();
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
            1u32,
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
            1u32,
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
