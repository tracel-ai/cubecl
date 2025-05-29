use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::global::GlobalWriter;
use crate::matmul::components::stage::StageEvent;
use crate::matmul::components::stage::StageToTileReader;
use crate::matmul::components::stage::shared::{CommonStageConfig, RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{NoEvent, StageBuffering, StageEventListener};
use crate::matmul::components::stage::{StageConfig, StageMatmul};
use crate::matmul::components::{Ident, global, tile};
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
    EP: StagePartitioner,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR, EP)>,
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
        let num_acc_n = config.tiling_dimensions(Ident::Rhs).tile_count_col() / n_acc_count;
        let start_m = m_acc_count * (SP::position() / num_acc_n);
        let start_n = n_acc_count * (SP::position() % num_acc_n);

        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                start_m,
                start_n,
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                config,
                listener,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
                start_m,
                start_n,
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragments,
                acc,
                config,
                listener,
            ),
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        let tmm_config = config.to_tmm_config();
        let mut lhs = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(config.tiling_scheme().tiles_in_partition_m()) {
            lhs.push(TMM::allocate_lhs(tmm_config));
        }

        let rhs = match config.buffering() {
            StageBuffering::Single => RhsTile::new_Single(TMM::allocate_rhs(tmm_config)),
            StageBuffering::Double => {
                RhsTile::new_Double((TMM::allocate_rhs(tmm_config), TMM::allocate_rhs(tmm_config)))
            }
        };

        (lhs, rhs)
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = stage_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;
        let m_iterations = global_config.tiling_scheme().tiles_in_partition_m();
        let n_iterations = global_config.tiling_scheme().tiles_in_partition_n();

        let mut out_smem = SharedMemory::<MP::EO>::new_lined(
            num_tile_lines * comptime!(SP::num_primitives(stage_config)),
            out_smem_line_size,
        );
        let slice_start = num_tile_lines * SP::position();
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

        let total_acc_n =
            stage_config.tiling_dimensions(Ident::Rhs).tile_count_col() / n_iterations;
        let m_offset = m_iterations * (SP::position() / total_acc_n);
        let n_offset = n_iterations * (SP::position() % total_acc_n);

        let mut m_iter = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_iterations] {
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_iterations] {
                let accumulator = Self::Accumulator::get_at(acc, m_iter, n_iter, stage_config);
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

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        Accumulators::<MP, TMM>::new(config)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        acc.zero(config);
    }

    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        acc.fill::<L>(loader, config);
    }

    fn init_writer(
        tensor: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SP::init_writer::<MP::EO>(tensor, x_offset, y_offset, batch_offset)
    }
}

#[cube]
impl<MP, TMM, RL, RR, EP> PartitionedStageMatmul<MP, TMM, RL, RR, EP>
where
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
    EP: StagePartitioner,
{
    /// Execute stage matmul with a single buffer for rhs.
    #[allow(clippy::too_many_arguments)]
    fn execute_single_buffer<SEL: StageEventListener>(
        start_m: u32,
        start_n: u32,
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Sequence<TMM::Lhs>,
        rhs_fragment: &mut TMM::Rhs,
        acc: &mut Accumulators<MP, TMM>,
        #[comptime] config: CommonStageConfig<TMM::Config>,
        mut listener: SEL,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = config.tiling_scheme().tiles_in_partition_m();
        let n_iterations = config.tiling_scheme().tiles_in_partition_n();
        let k_iterations = config.tiling_scheme().tiles_in_partition_k();

        let mut k_iter = comptime![0u32];
        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];
        let lhs_load_total = comptime!(m_iterations * k_iterations);
        let rhs_load_total = comptime!(n_iterations * k_iterations);
        let execute_total = comptime!(m_iterations * n_iterations * k_iterations);

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let mut m_iter = comptime![0u32];

            #[allow(clippy::explicit_counter_loop)]
            #[unroll]
            for _ in 0..m_iterations {
                let tile_lhs =
                    RL::read_tile::<TMM::Config>(lhs_reader, start_m + m_iter, k_iter, config);
                TMM::fill_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    config.to_tmm_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                );
                comptime!(lhs_load_counter += 1);

                comptime![m_iter += 1];
            }

            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..n_iterations {
                let rhs_tile_next =
                    RR::read_tile::<TMM::Config>(rhs_reader, k_iter, start_n + n_iter, config);
                TMM::fill_rhs(&rhs_tile_next, rhs_fragment, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }],
                );
                comptime!(rhs_load_counter += 1);

                let mut m_iter = comptime![0u32];

                #[allow(clippy::explicit_counter_loop)]
                #[unroll]
                for _ in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter, config);
                    TMM::execute(
                        lhs_fragment.index(m_iter),
                        rhs_fragment,
                        accumulator,
                        config.to_tmm_config(),
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime![StageEvent::TmmCompleted {
                            current: execute_counter,
                            total: execute_total
                        }],
                    );
                    comptime!(execute_counter += 1);

                    comptime![m_iter += 1];
                }

                comptime![n_iter += 1];
            }

            comptime![k_iter += 1];
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_double_buffer<SEL: StageEventListener>(
        start_m: u32,
        start_n: u32,
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Sequence<TMM::Lhs>,
        rhs_fragments: &mut (TMM::Rhs, TMM::Rhs),
        acc: &mut Accumulators<MP, TMM>,
        #[comptime] config: CommonStageConfig<TMM::Config>,
        mut listener: SEL,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = config.tiling_scheme().tiles_in_partition_m();
        let n_iterations = config.tiling_scheme().tiles_in_partition_n();
        let k_iterations = config.tiling_scheme().tiles_in_partition_k();

        let mut k_iter = comptime![0u32];

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];
        let lhs_load_total = comptime!(m_iterations * k_iterations);
        let rhs_load_total = comptime!(n_iterations * k_iterations);
        let execute_total = comptime!(m_iterations * n_iterations * k_iterations);

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let mut m_iter = comptime![0u32];

            #[allow(clippy::explicit_counter_loop)]
            #[unroll]
            for _ in 0..m_iterations {
                let tile_lhs =
                    RL::read_tile::<TMM::Config>(lhs_reader, start_m + m_iter, k_iter, config);
                TMM::fill_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    config.to_tmm_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                );
                comptime!(lhs_load_counter += 1);

                comptime![m_iter += 1];
            }

            let mut n_iter = comptime![0u32];

            let rhs_tile_first =
                RR::read_tile::<TMM::Config>(rhs_reader, k_iter, start_n + n_iter, config);
            TMM::fill_rhs(
                &rhs_tile_first,
                &mut rhs_fragments.0,
                config.to_tmm_config(),
            );
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: rhs_load_total
                }),
            );
            comptime!(rhs_load_counter += 1);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 1..n_iterations {
                let (current, next) = if comptime! {n_iter % 2 == 0} {
                    (&mut rhs_fragments.0, &mut rhs_fragments.1)
                } else {
                    (&mut rhs_fragments.1, &mut rhs_fragments.0)
                };

                let rhs_tile_next = RR::read_tile::<TMM::Config>(
                    rhs_reader,
                    k_iter,
                    start_n + comptime![n_iter + 1],
                    config,
                );
                TMM::fill_rhs(&rhs_tile_next, next, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }),
                );
                comptime!(rhs_load_counter += 1);

                let mut m_iter = comptime![0u32];

                #[allow(clippy::explicit_counter_loop)]
                #[unroll]
                for _ in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter, config);

                    TMM::execute(
                        lhs_fragment.index(m_iter),
                        current,
                        accumulator,
                        config.to_tmm_config(),
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime!(StageEvent::TmmCompleted {
                            current: execute_counter,
                            total: execute_total
                        }),
                    );
                    comptime!(execute_counter += 1);

                    comptime![m_iter += 1];
                }

                comptime![n_iter += 1];
            }

            let last = if comptime! {n_iter % 2 == 0} {
                &mut rhs_fragments.0
            } else {
                &mut rhs_fragments.1
            };

            let mut m_iter = comptime![0u32];

            #[allow(clippy::explicit_counter_loop)]
            #[unroll]
            for _ in 0..m_iterations {
                let accumulator = Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter, config);
                TMM::execute(
                    lhs_fragment.index(m_iter),
                    last,
                    accumulator,
                    config.to_tmm_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::TmmCompleted {
                        current: execute_counter,
                        total: execute_total
                    }),
                );
                comptime!(execute_counter += 1);

                comptime![m_iter += 1];
            }

            comptime![k_iter += 1];
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }
}
