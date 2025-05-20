use crate::matmul::components::global::GlobalWriter;
use crate::matmul::components::global::{AccumulatorLoader, UnitWriter};
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{NoEvent, StageBuffering, StageEvent, StageEventListener};
use crate::matmul::components::stage::{ReaderFamily, StageToTileReader};
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::tile::{TileMatmul, TileMatmulConfigInput};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem, global, tile};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::shared::StageVectorization;

pub struct UnitMatmulFamily<TMM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TMM, RF)>,
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for UnitMatmulFamily<TMM, RF> {
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    fn tile_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_shape
    }

    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        UnitMatmul<MP, TMM::Matmul<MP>, RF::Reader<MP::ES, TL>, RF::Reader<MP::ES, TR>>;
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> MatmulConfigFactory for UnitMatmulFamily<TMM, RF> {
    type Input = (
        CompleteStageTiling,
        StageBuffering,
        StageVectorization,
        (u32, u32),
    );
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_tmm = config.tiling_dimensions(Ident::Out).tile_count_row()
            * config.tiling_dimensions(Ident::Out).tile_count_col();
        let num_units = config.num_planes() * config.plane_dim();

        if num_tmm % num_units != 0 {
            return Err(Box::new(format!(
                "Error: at the moment, unit matmul must have num tile matmuls {:?} = num_units {:?}.",
                num_tmm, num_units
            )));
        }

        if num_tmm > 64 {
            return Err(Box::new(
                "Error: will probably bust shared memory".to_string(),
            ));
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
pub struct UnitMatmul<
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR)>,
}

#[cube]
impl<MP, TMM, RL, RR> StageMatmul<MP> for UnitMatmul<MP, TMM, RL, RR>
where
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = RL;
    type RhsReader = RR;
    type Accumulator = TMM::Accumulator;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;
    type Writer = UnitWriter<MP::EO>;

    fn execute(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_tile: &mut Self::LhsTile,
        rhs_tiles: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_reader,
            rhs_reader,
            lhs_tile,
            rhs_tiles,
            acc,
            config,
            NoEvent::new(),
        )
    }

    fn execute_with_listener<SEL: StageEventListener>(
        lhs_reader: &RL,
        rhs_reader: &RR,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        listener: SEL,
    ) {
        Self::execute_single_buffer::<SEL>(
            lhs_reader, rhs_reader, lhs_tile, rhs_tile, acc, config, listener,
        )
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        let tmm_config = config.to_tmm_config();
        let lhs = TMM::allocate_lhs(tmm_config);

        (lhs, TMM::allocate_rhs(tmm_config))
    }

    fn write_results<G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.to_smm_config().stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;
        let num_units = stage_config.num_planes() * stage_config.plane_dim();

        let mut out_smem =
            SharedMemory::<MP::EO>::new_lined(num_tile_lines * num_units, out_smem_line_size);
        let slice_start = num_tile_lines * UNIT_POS;
        let mut smem_slice = out_smem.slice_mut(slice_start, slice_start + num_tile_lines);

        let stage_n = stage_config.tiling_dimensions(Ident::Rhs).tile_count_col();
        let (m_index, n_index) = (UNIT_POS / stage_n, UNIT_POS % stage_n);

        TMM::write_results(acc, &mut smem_slice, stage_config.to_tmm_config());
        Self::Writer::write::<G>(out, smem_slice.to_slice(), m_index, n_index, global_config);
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        TMM::allocate_accumulator(config.to_tmm_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        TMM::zero_accumulator(acc, config.to_tmm_config());
    }

    fn fill_accumulator<L: AccumulatorLoader<MP>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        L::load::<TMM>(loader, acc, 0, config.to_tmm_config());
    }

    fn init_writer(
        tensor: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        Self::Writer::new(tensor, x_offset, y_offset, batch_offset)
    }
}

type Acc<MP, S> = <S as StageMatmul<MP>>::Accumulator;

#[cube]
impl<MP, TMM, RL, RR> UnitMatmul<MP, TMM, RL, RR>
where
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
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

        let k_iterations = config.tiling.tile_count.k;
        let stage_n = config.tiling.tile_count.n;

        let (m_index, n_index) = (UNIT_POS / stage_n, UNIT_POS % stage_n);

        let mut k_iter = comptime![0u32];

        let mut lhs_load_counter = comptime![0];
        let mut rhs_load_counter = comptime![0];
        let mut execute_counter = comptime![0];
        let lhs_load_total = comptime!(k_iterations);
        let rhs_load_total = comptime!(k_iterations);
        let execute_total = comptime!(k_iterations);

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let tile_lhs = RL::read_tile::<TMM::Config>(lhs_reader, m_index, k_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: lhs_load_counter,
                    total: lhs_load_total
                }],
            );
            comptime!(lhs_load_counter += 1);

            let tile_rhs = RR::read_tile::<TMM::Config>(rhs_reader, k_iter, n_index, config);
            TMM::fill_rhs(&tile_rhs, rhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::RhsLoaded {
                    current: rhs_load_counter,
                    total: rhs_load_total
                }],
            );
            comptime!(rhs_load_counter += 1);

            TMM::execute(lhs_fragment, rhs_fragment, acc, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::TmmCompleted {
                    current: execute_counter,
                    total: execute_total
                }],
            );
            comptime!(execute_counter += 1);

            comptime![k_iter += 1];
        }

        assert!(lhs_load_counter == lhs_load_total);
        assert!(rhs_load_counter == rhs_load_total);
        assert!(execute_counter == execute_total);
        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));
    }
}
