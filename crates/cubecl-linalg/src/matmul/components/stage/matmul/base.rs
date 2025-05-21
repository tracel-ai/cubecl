use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::stage::{StageConfig, StageToTileReader};
use crate::matmul::components::stage::{StageEvent, StageEventListener};
use crate::matmul::components::tile::TileMatmul;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use super::shared::{Accumulators, CommonStageConfig};

#[cube]
/// Execute stage matmul with a single buffer for rhs.
pub(crate) fn execute_single_buffer<
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
    SEL: StageEventListener,
>(
    lhs_reader: &RL,
    rhs_reader: &RR,
    lhs_fragment: &mut Sequence<TMM::Lhs>,
    rhs_fragment: &mut TMM::Rhs,
    acc: &mut Accumulators<MP, TMM>,
    #[comptime] config: CommonStageConfig<TMM::Config>,
    mut listener: SEL,
) {
    SEL::on_event(&mut listener, StageEvent::Begin);

    let (m_iterations, n_iterations) = acc.shape;
    let k_iterations = config.tiling.tile_count.k;

    let mut k_iter = comptime![0u32];

    let m_offset = UNIT_POS_Y * m_iterations;

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
                RL::read_tile::<TMM::Config>(lhs_reader, m_offset + m_iter, k_iter, config);
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
            let rhs_tile_next = RR::read_tile::<TMM::Config>(rhs_reader, k_iter, n_iter, config);
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
                let accumulator = Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter);
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

#[cube]
pub(crate) fn execute_double_buffer<
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
    SEL: StageEventListener,
>(
    lhs_reader: &RL,
    rhs_reader: &RR,
    lhs_fragment: &mut Sequence<TMM::Lhs>,
    rhs_fragments: &mut (TMM::Rhs, TMM::Rhs),
    acc: &mut Accumulators<MP, TMM>,
    #[comptime] config: CommonStageConfig<TMM::Config>,
    mut listener: SEL,
) {
    SEL::on_event(&mut listener, StageEvent::Begin);

    let (m_iterations, n_iterations) = acc.shape;
    let k_iterations = config.tiling.tile_count.k;

    let mut k_iter = comptime![0u32];
    let m_offset = UNIT_POS_Y * m_iterations;

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
                RL::read_tile::<TMM::Config>(lhs_reader, m_offset + m_iter, k_iter, config);
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

        let rhs_tile_first = RR::read_tile::<TMM::Config>(rhs_reader, k_iter, n_iter, config);
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

            let rhs_tile_next =
                RR::read_tile::<TMM::Config>(rhs_reader, k_iter, comptime![n_iter + 1], config);
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
                let accumulator = Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter);

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
            let accumulator = Accumulators::<MP, TMM>::get_at_mut(acc, m_iter, n_iter);
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
