use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulPrecision, MatrixPrecision, RhsG, RhsS, StageIdent,
    global::{
        GlobalConfig, GlobalWriter, PartitionedStage, PlaneWriter, SharedGlobalMatmulConfig,
        read::{FullStageGlobalReader, sync_full_cyclic},
    },
    stage::{RowMajorTilingOrder, StageConfig, StageMatmul, StridedStageMemory},
};
use cubecl_std::{
    CubeOption,
    tensor::{View, layout::Coords2d},
};

use crate::{
    components::{
        ConvolutionConfig,
        global::{
            ConvTilingLayout, GlobalConvolution,
            read::bias::{BiasGlobalReader, BiasStage},
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
            LhsStage = StridedStageMemory<LhsS<MP>, ConvTilingLayout>,
            RhsStage = StridedStageMemory<RhsS<MP>, ConvTilingLayout>,
            AccStage = BiasStage<AccS<MP>>,
            OutStage = PartitionedStage<AccS<MP>>,
        >,
{
    type LhsGlobalReader = FullStageGlobalReader<
        <MP::Lhs as MatrixPrecision>::Global,
        <MP::Lhs as MatrixPrecision>::Stage,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
    type Config = ConvolutionConfig<SharedGlobalMatmulConfig<SMM::Config>>;
    type RhsGlobalReader = FullStageGlobalReader<
        <MP::Rhs as MatrixPrecision>::Global,
        <MP::Rhs as MatrixPrecision>::Stage,
        sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
    >;
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

        acc_reader.load_stage::<SharedGlobalMatmulConfig<SMM::Config>>(config.matmul);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let mut barrier = ();

        for _ in 0..num_loops {
            sync_cube();

            lhs_reader.load_stage(&mut barrier, config.lhs_reader_config());
            rhs_reader.load_stage(&mut barrier, config.rhs_reader_config());

            sync_cube();

            SMM::execute(
                &lhs_reader.stage(),
                &rhs_reader.stage(),
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
                &partition_scheduler,
            );

            lhs_reader.advance_view();
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
        slice_size: Coords2d,
        _runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        Self::LhsGlobalReader::new(
            lhs.slice_unchecked(offset, slice_size),
            config.stage_config.elements_in_stage_k(),
            config.lhs_reader_config(),
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(
            rhs,
            config.stage_config.elements_in_stage_k(),
            config.rhs_reader_config(),
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
