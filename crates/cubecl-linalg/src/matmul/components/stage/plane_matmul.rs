use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::shared::{RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{NoEvent, StageBuffering, StageEvent, StageEventListener};
use crate::matmul::components::stage::{Reader, ReaderFamily};
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmul;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem, global, stage::StageWriter, tile};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

use super::shared::Accumulators;

pub struct PlaneMatmulFamily<TMM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TMM, RF)>,
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for PlaneMatmulFamily<TMM, RF> {
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        PlaneMatmul<MP, TMM::Matmul<MP>, RF::Reader<MP::ES, TL>, RF::Reader<MP::ES, TR>>;
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> MatmulConfigFactory for PlaneMatmulFamily<TMM, RF> {
    type Input = (CompleteStageTiling, StageBuffering);
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        check_num_planes(
            config.tiling_dimensions(Ident::Lhs).tile_count_row(),
            config.num_planes(),
        )?;
        TMM::check_config(&config.to_tmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tmm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let tile_shape = input.0.tile_shape;
        let tile_count = input.0.tile_count;

        let tmm_config = TMM::make_config(tile_shape, problem, cube_dim, cube_count, quantized);

        let tiling = CompleteStageTiling {
            tile_shape,
            tile_count,
        };

        CommonStageConfig::new(tmm_config, tiling, cube_dim.y, quantized, input.1)
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
        let tmm_config = config.to_tmm_config();
        let mut lhs = Sequence::new();
        lhs.push(TMM::allocate_lhs(tmm_config));

        (
            lhs,
            match config.buffering() {
                StageBuffering::Single => RhsTile::new_Single(TMM::allocate_rhs(tmm_config)),
                StageBuffering::Double => RhsTile::new_Double((
                    TMM::allocate_rhs(tmm_config),
                    TMM::allocate_rhs(tmm_config),
                )),
            },
        )
    }

    fn read_accumulator<SW: StageWriter<MP::EO>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;
        let shape_n = global_config.tiling_dimensions(Ident::Out).tile_count_col();

        let start = num_tile_lines * UNIT_POS_Y;

        let mut out_smem = SharedMemory::<MP::EO>::new_lined(
            num_tile_lines * stage_config.num_planes(),
            out_smem_line_size,
        );
        let mut smem_slice = out_smem.slice_mut(start, start + num_tile_lines);

        // TODO generalize shape
        let shape = (1u32, shape_n);

        // TODO iterate over m here

        let mut n_iter = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![shape.1] {
            let accumulator = Self::Accumulator::get_at(acc, 0u32, n_iter);
            TMM::read_accumulator(accumulator, &mut smem_slice, stage_config.to_tmm_config());
            SW::write::<MP::EO, G>(
                out,
                smem_slice.to_slice(),
                UNIT_POS_Y,
                n_iter,
                global_config,
            );

            comptime![n_iter += 1];
        }
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        // TODO generalize shape
        let shape = (1, config.tile_count().n);
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

fn check_num_planes(
    expected_num_planes: u32,
    actual_num_planes: u32,
) -> Result<(), InvalidConfigError> {
    if expected_num_planes != actual_num_planes {
        return Err(Box::new("Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}."));
    }

    Ok(())
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
        let k_iterations = comptime!(RL::num_k_iterations(config));
        assert!(m_iterations == 1, "Only m_iterations=1 supported for now");

        let acc_total_iterations = comptime![m_iterations * n_iterations * k_iterations];

        let mut k_iter = comptime![0u32];

        let lhs_fragment = lhs_fragment.index_mut(0);

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            // TODO add loop over m here

            let tile_lhs = RL::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, k_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: lhs_load_counter,
                    total: m_iterations * k_iterations
                }],
            );
            comptime!(lhs_load_counter += 1);

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
                        total: acc_total_iterations
                    }],
                );
                comptime!(rhs_load_counter += 1);

                let accumulator = Acc::<MP, Self>::get_at_mut(acc, 0u32, n_iter);
                TMM::execute(
                    lhs_fragment,
                    rhs_fragment,
                    accumulator,
                    config.to_tmm_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::TmmCompleted {
                        current: execute_counter,
                        total: acc_total_iterations
                    }],
                );
                comptime!(execute_counter += 1);

                comptime![n_iter += 1];
            }

            comptime![k_iter += 1];
        }

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
        assert!(m_iterations == 1, "Only m_iterations=1 supported for now");

        let k_iterations = comptime!(RL::num_k_iterations(config));
        let acc_total_iterations = comptime![k_iterations * n_iterations];

        let mut k_iter = comptime![0u32];
        let lhs_fragment = lhs_fragment.index_mut(0);

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let tile_lhs = RL::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, k_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: lhs_load_counter,
                    total: k_iterations
                }],
            );
            comptime!(lhs_load_counter += 1);

            let mut acc_iter = comptime![0u32];

            let rhs_tile_first = RR::read_tile::<TMM::Config>(rhs_reader, k_iter, acc_iter, config);
            TMM::fill_rhs(
                &rhs_tile_first,
                &mut rhs_fragments.0,
                config.to_tmm_config(),
            );
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: acc_total_iterations
                }),
            );
            comptime!(rhs_load_counter += 1);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 1..n_iterations {
                let (current, next) = if comptime! {acc_iter % 2 == 0} {
                    (&mut rhs_fragments.0, &mut rhs_fragments.1)
                } else {
                    (&mut rhs_fragments.1, &mut rhs_fragments.0)
                };

                let rhs_tile_next = RR::read_tile::<TMM::Config>(
                    rhs_reader,
                    k_iter,
                    comptime![acc_iter + 1],
                    config,
                );
                TMM::fill_rhs(&rhs_tile_next, next, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: rhs_load_counter,
                        total: acc_total_iterations
                    }),
                );
                comptime!(rhs_load_counter += 1);

                let accumulator = Acc::<MP, Self>::get_at_mut(acc, 0u32, acc_iter);

                TMM::execute(lhs_fragment, current, accumulator, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::TmmCompleted {
                        current: execute_counter,
                        total: acc_total_iterations
                    }),
                );
                comptime!(execute_counter += 1);

                comptime![acc_iter += 1];
            }

            let last = if comptime! {acc_iter % 2 == 0} {
                &mut rhs_fragments.0
            } else {
                &mut rhs_fragments.1
            };

            let accumulator = Acc::<MP, Self>::get_at_mut(acc, 0u32, acc_iter);
            TMM::execute(lhs_fragment, last, accumulator, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::TmmCompleted {
                    current: execute_counter,
                    total: acc_total_iterations
                }),
            );
            comptime!(execute_counter += 1);

            comptime![k_iter += 1];
        }

        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }
}
