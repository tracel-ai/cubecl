use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::shared::{RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{NoEvent, StageBuffering, StageEvent, StageEventListener};
use crate::matmul::components::stage::{Reader, ReaderFamily};
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::tile::{TileMatmul, TileMatmulConfigInput};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem, global, stage::Writer, tile};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

use super::shared::{Accumulators, StageVectorization};

pub struct PlaneMatmulFamily<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> {
    _phantom: PhantomData<(TMM, LRF, RRF)>,
}

impl<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> StageMatmulFamily
    for PlaneMatmulFamily<TMM, LRF, RRF>
{
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    fn tile_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_shape
    }

    type LhsReader = LRF;
    type RhsReader = RRF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        PlaneMatmul<MP, TMM::Matmul<MP>, LRF::Reader<MP::ES, TL>, RRF::Reader<MP::ES, TR>>;
}

impl<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> MatmulConfigFactory
    for PlaneMatmulFamily<TMM, LRF, RRF>
{
    type Input = (
        CompleteStageTiling,
        StageBuffering,
        StageVectorization,
        (u32, u32),
    );
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_rows = config.tiling_dimensions(Ident::Lhs).tile_count_row();
        let num_planes = config.num_planes();

        if num_rows % num_planes != 0 {
            return Err(Box::new(format!(
                "Error: Number of planes {num_planes} should divide number of rows {num_rows}."
            )));
        }

        TMM::check_config(&config.to_tmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tmm_config)
    }

    fn make_config(
        (tiling, buffering, vectorization, num_stages): Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let tile_shape = tiling.tile_shape;
        let tile_count = tiling.tile_count;

        let tile_input = TileMatmulConfigInput {
            vectorization,
            size: tile_shape,
        };
        let tmm_config = TMM::make_config(tile_input, problem, cube_dim, cube_count, quantized);

        let tiling = CompleteStageTiling {
            tile_shape,
            tile_count,
        };

        CommonStageConfig::new(
            tmm_config, tiling, cube_dim.y, quantized, buffering, num_stages,
        )
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct PlaneMatmul<
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: Reader<MP::ES>,
    RR: Reader<MP::ES>,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR)>,
}

#[cube]
impl<MP, TMM, RL, RR> StageMatmul<MP> for PlaneMatmul<MP, TMM, RL, RR>
where
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: Reader<MP::ES>,
    RR: Reader<MP::ES>,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = RL;
    type RhsReader = RR;
    type Accumulator = Accumulators<MP, TMM>;
    type LhsTile = Sequence<TMM::Lhs>;
    type RhsTile = RhsTile<TMM::Rhs>;

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
        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                config,
                listener,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
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
        let shape = (
            config.tile_count().m / config.num_planes(),
            config.tile_count().n,
        );

        let tmm_config = config.to_tmm_config();
        let mut lhs = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(shape.0) {
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

    fn read_accumulator<W: Writer<MP::EO>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut W,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.to_smm_config().stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;
        let (m_iterations, n_iterations) = acc.shape;

        let mut out_smem = SharedMemory::<MP::EO>::new_lined(
            num_tile_lines * stage_config.num_planes(),
            out_smem_line_size,
        );
        let slice_start = num_tile_lines * UNIT_POS_Y;
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

        let m_offset = UNIT_POS_Y * m_iterations;
        let mut m_iter = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_iterations] {
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_iterations] {
                let accumulator = Self::Accumulator::get_at(acc, m_iter, n_iter);
                TMM::read_accumulator(accumulator, &mut smem_slice, stage_config.to_tmm_config());
                W::write::<MP::EO, G>(
                    out,
                    smem_slice.to_slice(),
                    m_offset + m_iter,
                    n_iter,
                    global_config,
                );

                comptime![n_iter += 1];
            }
            comptime![m_iter += 1];
        }
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let shape = (
            config.tile_count().m / config.num_planes(),
            config.tile_count().n,
        );
        Accumulators::<MP, TMM>::new(shape, config)
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
}

type Acc<MP, S> = <S as StageMatmul<MP>>::Accumulator;

#[cube]
impl<MP, TMM, RL, RR> PlaneMatmul<MP, TMM, RL, RR>
where
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    RL: Reader<MP::ES>,
    RR: Reader<MP::ES>,
{
    // Execute stage matmul with a single buffer for rhs.
    fn execute_single_buffer<SEL: StageEventListener>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut <Self as StageMatmul<MP>>::LhsTile,
        rhs_fragment: &mut TMM::Rhs,
        acc: &mut Acc<MP, Self>,
        #[comptime] config: <Self as StageMatmul<MP>>::Config,
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
                let rhs_tile_next =
                    RR::read_tile::<TMM::Config>(rhs_reader, k_iter, n_iter, config);
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
                    let accumulator = Acc::<MP, Self>::get_at_mut(acc, m_iter, n_iter);
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

    // Execute stage matmul with two alternating buffers for rhs.
    fn execute_double_buffer<SEL: StageEventListener>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_fragment: &mut <Self as StageMatmul<MP>>::LhsTile,
        rhs_fragments: &mut (TMM::Rhs, TMM::Rhs),
        acc: &mut <Self as StageMatmul<MP>>::Accumulator,
        #[comptime] config: <Self as StageMatmul<MP>>::Config,
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
                    let accumulator = Acc::<MP, Self>::get_at_mut(acc, m_iter, n_iter);

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
                let accumulator = Acc::<MP, Self>::get_at_mut(acc, m_iter, n_iter);
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
