use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};
use cubecl_matmul::components::{
    AccG, AccS, LhsG, LhsS, MatmulIdent, MatmulPrecision, RhsG, RhsS,
    global::{
        GlobalConfig as _, GlobalWriter, PartitionedStage, PlaneWriter, read::arrive_tma,
        single_stage::tma::SimpleTmaConfig,
    },
    stage::{StageMatmul, StridedStage},
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
                bias::{BiasGlobalReader, BiasStage},
                im2col_tma::{TmaIm2colGlobalReader, TmaIm2colTiling},
                weight_tma::{TmaWeightGlobalReader, TmaWeightTiling},
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
            LhsStage = StridedStage<LhsS<MP>, TmaIm2colTiling>,
            RhsStage = StridedStage<RhsS<MP>, TmaWeightTiling>,
            AccStage = BiasStage<AccS<MP>>,
            OutStage = PartitionedStage<AccS<MP>>,
        >,
{
    type Config = ConvolutionConfig<SimpleTmaConfig<SMM::Config>>;

    type LhsGlobalReader = TmaIm2colGlobalReader<MP::Lhs, Self::Config>;
    type RhsGlobalReader = TmaWeightGlobalReader<MP::Rhs, SMM::Config>;
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
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = range.div_ceil(k_step);

        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let stage_bytes_lhs =
            comptime!(config.tiling_scheme().elements_in_stage_mk() * lhs_elem_size);
        let stage_bytes_rhs =
            comptime!(config.tiling_scheme().elements_in_stage_nk() * rhs_elem_size);
        let stages_bytes = stage_bytes_lhs + stage_bytes_rhs;

        acc_reader.load_stage::<Self::Config>(config);
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        let partition_scheduler = SMM::init_scheduler(config.stage_config());

        sync_cube();

        SMM::load_accumulators(&acc_reader.stage(), acc, config.stage_config());

        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            lhs_reader.fill_stage(&barrier, 0u32);
            rhs_reader.fill_stage(&barrier, 0u32, config.stage_config());

            arrive_tma(&barrier, stages_bytes);

            barrier.wait();

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
            rhs_reader.advance_view(k_step);
        }

        sync_cube();

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
        lhs: VirtualTensor<LhsG<MP>>,
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader {
        let (x_offset, y_offset) = offset;
        Self::LhsGlobalReader::new(lhs, x_offset, y_offset, runtime_args, 1u32, config)
    }

    fn init_rhs_global_reader(
        rhs: VirtualTensor<RhsG<MP>>,
        offset: Coords2d,
        _slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader {
        let (x_offset, y_offset) = offset;
        Self::RhsGlobalReader::new(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            runtime_args,
            1u32,
            config.stage_memory_config(MatmulIdent::Rhs),
        )
    }

    fn init_bias_global_reader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        slice_size: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader {
        Self::AccGlobalReader::new(
            bias,
            n_offset,
            slice_size,
            config.stage_memory_config(MatmulIdent::Out),
        )
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
        Self::GlobalWriter::new::<SMM::Config>(
            out.slice_mut_unchecked(offset, slice_size),
            global_conf,
            config.stage_config(),
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators {
        SMM::init_accumulators(config.stage_config())
    }
}
