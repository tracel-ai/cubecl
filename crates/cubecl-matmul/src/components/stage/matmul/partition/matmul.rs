use std::marker::PhantomData;

use super::fragments::{Accumulators, RhsTile, RhsTileExpand};
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::matmul::scheduler::PartitionScheduler;
use crate::components::stage::{PartitionBuffering, StageEventListener};
use crate::components::stage::{PartitionSchedulerScheme, StageMemoryConfig};
use crate::components::tile::{TileConfig, TileMatmul};
use crate::components::{AccS, stage::StageEvent};
use crate::components::{LhsS, MatmulPrecision, PartitionSize, RhsS, StageSize};
use crate::components::{MatrixPrecision, stage::Stage};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedPartitionMatmulConfig<TC: TileConfig> {
    pub tile_config: TC,
    pub partition_size: PartitionSize,
    pub partition_buffering: PartitionBuffering,
    pub plane_role_config: PlaneRoleConfig,
    pub plane_dim: u32,
    pub stage_size: StageSize,
    pub partition_schedule_scheme: PartitionSchedulerScheme,
    pub lhs_smem_config: StageMemoryConfig,
    pub rhs_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

impl<TC: TileConfig> SharedPartitionMatmulConfig<TC> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tile_config: TC,
        partition_size: PartitionSize,
        partition_buffering: PartitionBuffering,
        plane_role_config: PlaneRoleConfig,
        plane_dim: u32,
        stage_size: StageSize,
        partition_schedule_scheme: PartitionSchedulerScheme,
        lhs_smem_config: StageMemoryConfig,
        rhs_smem_config: StageMemoryConfig,
        out_smem_config: StageMemoryConfig,
    ) -> Self {
        Self {
            tile_config,
            partition_size,
            partition_buffering,
            plane_role_config,
            plane_dim,
            stage_size,
            partition_schedule_scheme,
            lhs_smem_config,
            rhs_smem_config,
            out_smem_config,
        }
    }
}

/// Matmul for a whole partition, a region of the Stage Matmul
/// executed by a single compute primitive (unit or plane)
pub struct PartitionMatmul<
    MP: MatmulPrecision,
    TMM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    StageLhs: Stage<LhsS<MP>, ReadOnly, TileKind = TMM::LhsTile>,
    StageRhs: Stage<RhsS<MP>, ReadOnly, TileKind = TMM::RhsTile>,
    StageAcc: Stage<AccS<MP>, ReadOnly, TileKind = TMM::AccTile>,
> {
    _phantom: PhantomData<(MP, TMM, StageLhs, StageRhs, StageAcc)>,
}

#[cube]
impl<MP, TM, StageLhs, StageRhs, StageAcc> PartitionMatmul<MP, TM, StageLhs, StageRhs, StageAcc>
where
    MP: MatmulPrecision,
    TM: TileMatmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
    StageLhs: Stage<LhsS<MP>, ReadOnly, TileKind = TM::LhsTile>,
    StageRhs: Stage<RhsS<MP>, ReadOnly, TileKind = TM::RhsTile>,
    StageAcc: Stage<AccS<MP>, ReadOnly, TileKind = TM::AccTile>,
{
    #[allow(clippy::too_many_arguments)]
    /// Execute all Tile Matmuls inside the partition
    /// Can be with single or double buffering
    pub fn execute_with_listener<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<TM::LhsFragment>,
        rhs_fragments: &mut RhsTile<TM::RhsFragment>,
        acc: &mut Accumulators<MP, TM>,
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
        listener: SEL,
        partition_iterator: &PartitionScheduler,
    ) {
        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                lhs_stage,
                rhs_stage,
                lhs_fragment,
                rhs_fragment,
                acc,
                shared_config,
                listener,
                partition_iterator,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
                lhs_stage,
                rhs_stage,
                lhs_fragment,
                rhs_fragments,
                acc,
                shared_config,
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
    /// Make sure to load inputs before execution.
    pub fn init_tile_inputs(
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
    ) -> (Sequence<TM::LhsFragment>, RhsTile<TM::RhsFragment>) {
        let mut lhs = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(shared_config.partition_size.m()) {
            lhs.push(TM::allocate_lhs(
                shared_config.lhs_smem_config.matrix_layout,
                shared_config.tile_config,
            ));
        }

        let rhs = match shared_config.partition_buffering {
            PartitionBuffering::Single => RhsTile::new_Single(TM::allocate_rhs(
                shared_config.rhs_smem_config.matrix_layout,
                shared_config.tile_config,
            )),
            PartitionBuffering::Double => RhsTile::new_Double((
                TM::allocate_rhs(
                    shared_config.rhs_smem_config.matrix_layout,
                    shared_config.tile_config,
                ),
                TM::allocate_rhs(
                    shared_config.rhs_smem_config.matrix_layout,
                    shared_config.tile_config,
                ),
            )),
        };

        (lhs, rhs)
    }

    /// Initialize accumulators
    ///     
    /// # Safety
    ///
    /// This may point towards uninitialized memory.
    /// Make sure to call `load_accumulator` prior to execute_with_listener.
    pub fn init_accumulator(
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
    ) -> Accumulators<MP, TM> {
        Accumulators::<MP, TM>::new(
            shared_config.partition_size,
            shared_config.out_smem_config.matrix_layout,
            shared_config.tile_config,
        )
    }

    /// Fill accumulators through a stage
    pub fn load_accumulator(
        stage: &StageAcc,
        acc: &mut Accumulators<MP, TM>,
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
    ) {
        acc.load::<StageAcc>(
            stage,
            shared_config.partition_size.m(),
            shared_config.partition_size.n(),
            shared_config.tile_config,
        );
    }

    /// Execute partition matmul with a single buffer for rhs.
    ///
    /// This function can call functions at various events through the listener.
    #[allow(clippy::too_many_arguments)]
    fn execute_single_buffer<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<TM::LhsFragment>,
        rhs_fragment: &mut TM::RhsFragment,
        acc: &mut Accumulators<MP, TM>,
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = shared_config.partition_size.m();
        let n_iterations = shared_config.partition_size.n();
        let k_iterations = shared_config.partition_size.k();

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];
        let lhs_load_total = comptime!(m_iterations * k_iterations);
        let rhs_load_total = comptime!(n_iterations * k_iterations);
        let execute_total = comptime!(m_iterations * n_iterations * k_iterations);

        #[unroll]
        for k_iter in 0..k_iterations {
            let k_load_iter = partition_scheduler.map_k(k_iter);

            #[unroll]
            for m_iter in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter);

                let tile_lhs = StageLhs::tile(lhs_stage, (m_load_iter, k_load_iter));
                TM::load_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    shared_config.tile_config,
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                );
                comptime!(lhs_load_counter += 1);
            }

            #[unroll]
            for n_iter in 0..n_iterations {
                let n_load_iter = partition_scheduler.map_n(n_iter);

                let rhs_tile_next = StageRhs::tile(rhs_stage, (k_load_iter, n_load_iter));
                TM::load_rhs(&rhs_tile_next, rhs_fragment, shared_config.tile_config);
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }],
                );
                comptime!(rhs_load_counter += 1);

                #[unroll]
                for m_iter in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TM>::get_at_mut(acc, m_iter, n_iter, n_iterations);
                    TM::execute(
                        lhs_fragment.index(m_iter),
                        rhs_fragment,
                        accumulator,
                        shared_config.tile_config,
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime![StageEvent::TileMatmulCompleted {
                            current: execute_counter,
                            total: execute_total
                        }],
                    );
                    comptime!(execute_counter += 1);
                }
            }
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }

    #[allow(clippy::too_many_arguments)]
    /// Execute partition matmul with a double buffering for rhs.
    ///
    /// This function can call functions at various events through the listener.
    fn execute_double_buffer<SEL: StageEventListener>(
        lhs_stage: &StageLhs,
        rhs_stage: &StageRhs,
        lhs_fragment: &mut Sequence<TM::LhsFragment>,
        rhs_fragments: &mut (TM::RhsFragment, TM::RhsFragment),
        acc: &mut Accumulators<MP, TM>,
        #[comptime] shared_config: SharedPartitionMatmulConfig<TM::Config>,
        mut listener: SEL,
        partition_scheduler: &PartitionScheduler,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let m_iterations = shared_config.partition_size.m();
        let n_iterations = shared_config.partition_size.n();
        let k_iterations = shared_config.partition_size.k();

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];
        let lhs_load_total = comptime!(m_iterations * k_iterations);
        let rhs_load_total = comptime!(n_iterations * k_iterations);
        let execute_total = comptime!(m_iterations * n_iterations * k_iterations);

        #[unroll]
        for k_iter in 0..k_iterations {
            let k_load_iter = partition_scheduler.map_k(k_iter);

            #[unroll]
            for m_iter in 0..m_iterations {
                let m_load_iter = partition_scheduler.map_m(m_iter);

                let tile_lhs = StageLhs::tile(lhs_stage, (m_load_iter, k_load_iter));
                TM::load_lhs(
                    &tile_lhs,
                    lhs_fragment.index_mut(m_iter),
                    shared_config.tile_config,
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::LhsLoaded {
                        current: lhs_load_counter,
                        total: lhs_load_total
                    }],
                );
                comptime!(lhs_load_counter += 1);
            }

            let mut n_iter = comptime![0u32];
            let n_load_iter = partition_scheduler.map_n(n_iter);

            let rhs_tile_first = StageRhs::tile(rhs_stage, (k_load_iter, n_load_iter));
            TM::load_rhs(
                &rhs_tile_first,
                &mut rhs_fragments.0,
                shared_config.tile_config,
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

                let n_load_iter = partition_scheduler.map_n(comptime![n_iter + 1]);
                let rhs_tile_next = StageRhs::tile(rhs_stage, (k_load_iter, n_load_iter));
                TM::load_rhs(&rhs_tile_next, next, shared_config.tile_config);
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: rhs_load_total
                    }),
                );
                comptime!(rhs_load_counter += 1);

                #[unroll]
                for m_iter in 0..m_iterations {
                    let accumulator =
                        Accumulators::<MP, TM>::get_at_mut(acc, m_iter, n_iter, n_iterations);

                    TM::execute(
                        lhs_fragment.index(m_iter),
                        current,
                        accumulator,
                        shared_config.tile_config,
                    );
                    SEL::on_event(
                        &mut listener,
                        comptime!(StageEvent::TileMatmulCompleted {
                            current: execute_counter,
                            total: execute_total
                        }),
                    );
                    comptime!(execute_counter += 1);
                }

                comptime![n_iter += 1];
            }

            let last = if comptime! {n_iter % 2 == 0} {
                &mut rhs_fragments.0
            } else {
                &mut rhs_fragments.1
            };

            #[unroll]
            for m_iter in 0..m_iterations {
                let accumulator =
                    Accumulators::<MP, TM>::get_at_mut(acc, m_iter, n_iter, n_iterations);
                TM::execute(
                    lhs_fragment.index(m_iter),
                    last,
                    accumulator,
                    shared_config.tile_config,
                );
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::TileMatmulCompleted {
                        current: execute_counter,
                        total: execute_total
                    }),
                );
                comptime!(execute_counter += 1);
            }
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }
}
