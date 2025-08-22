use std::marker::PhantomData;

use super::fragments::{Accumulators, RhsTile, RhsTileExpand};
use crate::components::InputPrecision;
use crate::components::global::AccumulatorLoader;
use crate::components::stage::StageConfig;
use crate::components::stage::StageEvent;
use crate::components::stage::StageToTileReader;
use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::stage::{PartitionBuffering, StageEventListener};
use crate::components::tile::TileMatmul;
use crate::components::{LhsS, MatmulPrecision, RhsS};
use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Matmul for a whole partition, a region of the Stage Matmul
/// executed by a single compute primitive (unit or plane)
pub struct PartitionMatmul<
    MP: MatmulPrecision,
    TMM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            MP::EA,
        >,
    RL: StageToTileReader<LhsS<MP>>,
    RR: StageToTileReader<RhsS<MP>>,
    S: StageConfig,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR, S)>,
}

#[cube]
impl<MP, TM, RL, RR, S> PartitionMatmul<MP, TM, RL, RR, S>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            MP::EA,
        >,
    RL: StageToTileReader<LhsS<MP>>,
    RR: StageToTileReader<RhsS<MP>>,
    S: StageConfig<TileConfig = TM::Config>,
{
    #[allow(clippy::too_many_arguments)]
    /// Execute all Tile Matmuls inside the partition
    /// Can be with single or double buffering
    pub fn execute_with_listener<SEL: StageEventListener<S>>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Sequence<TM::Lhs>,
        rhs_fragments: &mut RhsTile<TM::Rhs>,
        acc: &mut Accumulators<MP, TM, S>,
        #[comptime] config: S,
        listener: SEL,
        partition_iterator: &PartitionScheduler,
    ) {
        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                config,
                listener,
                partition_iterator,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragments,
                acc,
                config,
                listener,
                partition_iterator,
            ),
        }
    }

    /// Initialize Lhs and Rhs inputs
    ///
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to fill inputs before execution.
    pub fn init_tile_inputs(#[comptime] config: S) -> (Sequence<TM::Lhs>, RhsTile<TM::Rhs>) {
        let tile_config = config.tile_config();
        let mut lhs = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(config.tiling_scheme().tiles_in_stage_partition_m()) {
            lhs.push(TM::allocate_lhs(tile_config));
        }

        let rhs = match config.partition_buffering() {
            PartitionBuffering::Single => RhsTile::new_Single(TM::allocate_rhs(tile_config)),
            PartitionBuffering::Double => {
                RhsTile::new_Double((TM::allocate_rhs(tile_config), TM::allocate_rhs(tile_config)))
            }
        };

        (lhs, rhs)
    }

    /// Initialize accumulators
    ///     
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call zero_accumulator or fill_accumulator prior to execute_with_listener.
    pub fn init_accumulator(#[comptime] config: S) -> Accumulators<MP, TM, S> {
        Accumulators::<MP, TM, S>::new(config)
    }

    /// Fill accumulators with zeroes
    pub fn zero_accumulator(acc: &mut Accumulators<MP, TM, S>, #[comptime] config: S) {
        acc.zero(config);
    }

    /// Fill accumulators through an AccumulatorLoader
    pub fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Accumulators<MP, TM, S>,
        #[comptime] config: S,
    ) {
        acc.fill::<L>(loader, config);
    }

    /// Execute partition matmul with a single buffer for rhs.
    ///
    /// This function can call functions at various events through the listener.
    #[allow(clippy::too_many_arguments)]
    fn execute_single_buffer<SEL: StageEventListener<S>>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Sequence<TM::Lhs>,
        rhs_fragment: &mut TM::Rhs,
        acc: &mut Accumulators<MP, TM, S>,
        #[comptime] config: S,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin, config);

        let m_iterations = config.tiling_scheme().tiles_in_stage_partition_m();
        let n_iterations = config.tiling_scheme().tiles_in_stage_partition_n();
        let k_iterations = config.tiling_scheme().tiles_in_stage_partition_k();

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
            let k_load_iter = partition_scheduler.map_k(k_iter);

            #[allow(clippy::explicit_counter_loop)]
            #[unroll]
            for _ in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter);

                let tile_lhs = RL::read_tile::<S::StageMemoryConfig>(
                    lhs_reader,
                    m_load_iter,
                    k_load_iter,
                    config.stage_memory_config(),
                );
                TM::fill_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    config.tile_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                    config,
                );
                comptime!(lhs_load_counter += 1);

                comptime![m_iter += 1];
            }

            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..n_iterations {
                let n_load_iter = partition_scheduler.map_n(n_iter);

                let rhs_tile_next = RR::read_tile::<S::StageMemoryConfig>(
                    rhs_reader,
                    k_load_iter,
                    n_load_iter,
                    config.stage_memory_config(),
                );
                TM::fill_rhs(&rhs_tile_next, rhs_fragment, config.tile_config());
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }],
                    config,
                );
                comptime!(rhs_load_counter += 1);

                let mut m_iter = comptime![0u32];

                #[allow(clippy::explicit_counter_loop)]
                #[unroll]
                for _ in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TM, S>::get_at_mut(acc, m_iter, n_iter, config);
                    TM::execute(
                        lhs_fragment.index(m_iter),
                        rhs_fragment,
                        accumulator,
                        config.tile_config(),
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime![StageEvent::TileMatmulCompleted {
                            current: execute_counter,
                            total: execute_total
                        }],
                        config,
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
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish), config);
    }

    #[allow(clippy::too_many_arguments)]
    /// Execute partition matmul with a double buffering for rhs.
    ///
    /// This function can call functions at various events through the listener.
    fn execute_double_buffer<SEL: StageEventListener<S>>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut Sequence<TM::Lhs>,
        rhs_fragments: &mut (TM::Rhs, TM::Rhs),
        acc: &mut Accumulators<MP, TM, S>,
        #[comptime] config: S,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin, config);

        let m_iterations = config.tiling_scheme().tiles_in_stage_partition_m();
        let n_iterations = config.tiling_scheme().tiles_in_stage_partition_n();
        let k_iterations = config.tiling_scheme().tiles_in_stage_partition_k();

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
            let k_load_iter = partition_scheduler.map_k(k_iter);

            #[allow(clippy::explicit_counter_loop)]
            #[unroll]
            for _ in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter);

                let tile_lhs = RL::read_tile::<S::StageMemoryConfig>(
                    lhs_reader,
                    m_load_iter,
                    k_load_iter,
                    config.stage_memory_config(),
                );
                TM::fill_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    config.tile_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                    config,
                );
                comptime!(lhs_load_counter += 1);

                comptime![m_iter += 1];
            }

            let mut n_iter = comptime![0u32];
            let n_load_iter = partition_scheduler.map_n(n_iter);

            let rhs_tile_first = RR::read_tile::<S::StageMemoryConfig>(
                rhs_reader,
                k_load_iter,
                n_load_iter,
                config.stage_memory_config(),
            );
            TM::fill_rhs(&rhs_tile_first, &mut rhs_fragments.0, config.tile_config());
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: rhs_load_total
                }),
                config,
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

                let n_load_iter = partition_scheduler.map_n(comptime![n_iter + 1]);
                let rhs_tile_next = RR::read_tile::<S::StageMemoryConfig>(
                    rhs_reader,
                    k_load_iter,
                    n_load_iter,
                    config.stage_memory_config(),
                );
                TM::fill_rhs(&rhs_tile_next, next, config.tile_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }),
                    config,
                );
                comptime!(rhs_load_counter += 1);

                let mut m_iter = comptime![0u32];

                #[allow(clippy::explicit_counter_loop)]
                #[unroll]
                for _ in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TM, S>::get_at_mut(acc, m_iter, n_iter, config);

                    TM::execute(
                        lhs_fragment.index(m_iter),
                        current,
                        accumulator,
                        config.tile_config(),
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime!(StageEvent::TileMatmulCompleted {
                            current: execute_counter,
                            total: execute_total
                        }),
                        config,
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
                let accumulator =
                    Accumulators::<MP, TM, S>::get_at_mut(acc, m_iter, n_iter, config);
                TM::execute(
                    lhs_fragment.index(m_iter),
                    last,
                    accumulator,
                    config.tile_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::TileMatmulCompleted {
                        current: execute_counter,
                        total: execute_total
                    }),
                    config,
                );
                comptime!(execute_counter += 1);

                comptime![m_iter += 1];
            }

            comptime![k_iter += 1];
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish), config);
    }
}
