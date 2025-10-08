use crate::components::LhsG;
use crate::components::global::GlobalMatmul;
use crate::components::global::read::TmaGlobalReader;
use crate::components::global::read::arrive_tma;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::stage::StageMatmul;
use crate::components::{AccG, RhsG};
use crate::components::{AccS, MatmulIdent};
use crate::components::{LhsS, global::read::TmaStage, stage::FilledStage};
use crate::components::{MatmulPrecision, global::read::ZeroGlobalReader};
use crate::components::{RhsS, global::GlobalWriter};
use barrier::Barrier;
use cubecl_core::prelude::{barrier::BarrierLevel, *};
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::View;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::layout::Coords2d};
use std::marker::PhantomData;

use crate::components::global::GlobalConfig;

/// Performs matrix multiplication at the global level
/// Similar to simple matmul but using tma loading
pub struct SimpleTmaMatmul<MP: MatmulPrecision, SMM: StageMatmul<MP>, GW: GlobalWriter<MP::Acc>> {
    _phantom: PhantomData<(MP, SMM, GW)>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, GW> GlobalMatmul<MP> for SimpleTmaMatmul<MP, SMM, GW>
where
    SMM: StageMatmul<
            MP,
            LhsStage = TmaStage<MP::Lhs>,
            RhsStage = TmaStage<MP::Rhs>,
            AccStage = FilledStage<AccS<MP>>,
            OutStage = GW::Stage,
        >,
    GW: GlobalWriter<MP::Acc>,
{
    type Config = SimpleTmaConfig<SMM::Config>;
    type LhsGlobalReader = TmaGlobalReader<MP::Lhs>;
    type RhsGlobalReader = TmaGlobalReader<MP::Rhs>;
    type AccGlobalReader = ZeroGlobalReader<MP::Acc>;
    type GlobalWriter = GW;
    type Accumulators = SMM::Accumulators;

    fn execute(
        mut lhs_reader: Self::LhsGlobalReader,
        mut rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        mut out_writer: Self::GlobalWriter,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let num_bytes_lhs =
            comptime!(config.tiling_scheme().elements_in_stage_mk() * lhs_elem_size);
        let num_bytes_rhs =
            comptime!(config.tiling_scheme().elements_in_stage_nk() * rhs_elem_size);
        let num_bytes_stages = num_bytes_lhs + num_bytes_rhs;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            // Start loading
            lhs_reader.load_stage(&barrier);
            rhs_reader.load_stage(&barrier);

            arrive_tma(&barrier, num_bytes_stages);

            barrier.wait();

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

        let mut out_stage = Self::GlobalWriter::stage(&out_writer);

        SMM::write_results::<Self::GlobalWriter, Self::Config>(
            acc,
            &mut out_stage,
            &mut out_writer,
            &partition_scheduler,
            config.stage_config(),
            config,
        );
    }

    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        Self::LhsGlobalReader::new(
            lhs,
            config.k_step,
            MatmulIdent::Lhs,
            config.stage_memory_config(MatmulIdent::Lhs),
        )
    }

    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        Self::RhsGlobalReader::new(
            rhs,
            config.k_step,
            MatmulIdent::Rhs,
            config.stage_memory_config(MatmulIdent::Rhs),
        )
    }

    fn init_acc_global_reader(
        acc: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        #[comptime] _config: Self::Config,
    ) -> Self::AccGlobalReader {
        match acc {
            CubeOption::None => ZeroGlobalReader::new(),
            CubeOption::Some(_) => panic!("Accumulator loading is not yet supported"),
        }
    }

    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter {
        let conf = config.global_memory_config(MatmulIdent::Out);
        Self::GlobalWriter::init::<SMM::Config>(out, conf, config.stage_config())
    }

    fn init_accumulators(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
