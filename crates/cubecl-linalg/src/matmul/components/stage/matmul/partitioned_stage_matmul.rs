use crate::matmul::components::Ident;
use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::global::GlobalWriter;
use crate::matmul::components::stage::StageConfig;
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::stage::StageToTileReader;
use crate::matmul::components::stage::matmul::partition_matmul::PartitionMatmul;
use crate::matmul::components::stage::shared::{CommonStageConfig, RhsTile};
use crate::matmul::components::stage::{NoEvent, StageEventListener};
use crate::matmul::components::{global, tile};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::shared::Accumulators;

#[cube]
pub trait StagePartitioner: Send + Sync + 'static {
    type Writer<EO: Numeric>: GlobalWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO>;

    fn position() -> u32;

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32);
}

pub struct PartitionedStageMatmul<
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
    SP: StagePartitioner,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR, SP)>,
}

#[cube]
impl<MP, TMM, RL, RR, SP> StageMatmul<MP> for PartitionedStageMatmul<MP, TMM, RL, RR, SP>
where
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
    SP: StagePartitioner,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = RL;
    type RhsReader = RR;
    type Accumulator = Accumulators<MP, TMM>;
    type LhsTile = Sequence<TMM::Lhs>;
    type RhsTile = RhsTile<TMM::Rhs>;
    type Writer = SP::Writer<MP::EO>;

    fn execute(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            NoEvent::new(),
        )
    }

    fn execute_with_listener<SEL: StageEventListener>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        listener: SEL,
    ) {
        let m_acc_count = config.tiling_scheme().tiles_in_partition_m();
        let n_acc_count = config.tiling_scheme().tiles_in_partition_n();
        let num_partitions_n = config.tiling_scheme().partitions_in_stage_n();
        let start_m = m_acc_count * (SP::position() / num_partitions_n);
        let start_n = n_acc_count * (SP::position() % num_partitions_n);

        PartitionMatmul::<MP, TMM, RL, RR>::execute_with_listener::<SEL>(
            start_m,
            start_n,
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            config,
            listener,
        );
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        PartitionMatmul::<MP, TMM, RL, RR>::init_tile_inputs(config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        PartitionMatmul::<MP, TMM, RL, RR>::init_accumulator(config)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        PartitionMatmul::<MP, TMM, RL, RR>::zero_accumulator(acc, config);
    }

    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        PartitionMatmul::<MP, TMM, RL, RR>::fill_accumulator::<L>(loader, acc, config);
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Accumulators<MP, TMM>,
        out: &mut Self::Writer,
        #[comptime] stage_config: CommonStageConfig<TMM::Config>,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = stage_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_scheme().elements_in_tile_mn() / out_smem_line_size;
        let m_iterations = global_config.tiling_scheme().tiles_in_partition_m();
        let n_iterations = global_config.tiling_scheme().tiles_in_partition_n();

        let mut out_smem = SharedMemory::<MP::EO>::new_lined(
            num_tile_lines * comptime!(SP::num_primitives(stage_config)),
            out_smem_line_size,
        );
        let slice_start = num_tile_lines * SP::position();
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

        let num_partitions_n = stage_config.tiling_scheme().partitions_in_stage_n();
        let m_offset = m_iterations * (SP::position() / num_partitions_n);
        let n_offset = n_iterations * (SP::position() % num_partitions_n);

        let mut m_iter = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_iterations] {
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_iterations] {
                let accumulator =
                    Accumulators::<MP, TMM>::get_at(acc, m_iter, n_iter, stage_config);
                TMM::write_results(accumulator, &mut smem_slice, stage_config.to_tmm_config());
                Self::Writer::write::<G>(
                    out,
                    smem_slice.to_slice(),
                    m_offset + m_iter,
                    n_offset + n_iter,
                    global_config,
                );

                comptime![n_iter += 1];
            }
            comptime![m_iter += 1];
        }
    }

    fn init_writer(
        tensor: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SP::init_writer(tensor, x_offset, y_offset, batch_offset)
    }
}
