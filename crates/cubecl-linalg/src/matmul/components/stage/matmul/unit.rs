use crate::matmul::components::global::GlobalWriter;
use crate::matmul::components::global::{AccumulatorLoader, UnitWriter};
use crate::matmul::components::stage::matmul::base::execute_single_buffer;
use crate::matmul::components::stage::shared::{CommonStageConfig, RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{
    NoEvent, StageBuffering, StageEventListener, StageVectorization,
};
use crate::matmul::components::stage::{ReaderFamily, StageToTileReader};
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulConfigInput;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
    MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem, global, tile};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::shared::Accumulators;

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
        (u32, u32),
    );
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_acc = config.tiling_dimensions(Ident::Out).tile_count();
        // TODO when accumulator shape is a struct, implement a num_elems
        let acc_per_unit = config.accumulator_shape().0 * config.accumulator_shape().1;

        if num_acc % acc_per_unit != 0 {
            return Err(Box::new(format!(
                "Error: Number of accumulators {num_acc} should be divisible by number of accumulators per unit {acc_per_unit}."
            )));
        }

        let num_units_needed = num_acc / acc_per_unit;
        let num_units = config.plane_dim() * config.num_planes();

        if num_units != num_units_needed {
            return Err(Box::new(format!(
                "Error: Number of units {num_units} should be {num_units_needed}."
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
        (tiling, buffering, vectorization, num_stages, acc_shape): Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
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
        let tmm_config = TMM::make_config(
            tile_input, problem, line_sizes, cube_dim, cube_count, quantized,
        );

        let tiling = CompleteStageTiling {
            tile_shape,
            tile_count,
        };

        CommonStageConfig::new(
            tmm_config, tiling, cube_dim.y, quantized, buffering, num_stages, acc_shape,
        )
    }
}

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
    type Accumulator = Accumulators<MP, TMM>;
    type LhsTile = Sequence<TMM::Lhs>;
    type RhsTile = RhsTile<TMM::Rhs>;
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
        lhs_tiles: &mut Self::LhsTile,
        rhs_tiles: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
        listener: SEL,
    ) {
        // TODO support double buffer

        // TODO this is duplicated in write
        let (m_acc_shape, n_acc_shape) = config.accumulator_shape();
        let num_acc_n = config.tiling_dimensions(Ident::Rhs).tile_count_col() / n_acc_shape;
        let start_m = m_acc_shape * (UNIT_POS / num_acc_n);
        let start_n = n_acc_shape * (UNIT_POS % num_acc_n);

        match rhs_tiles {
            RhsTile::Single(rhs_tile) => execute_single_buffer::<MP, TMM, RL, RR, SEL>(
                start_m, start_n, lhs_reader, rhs_reader, lhs_tiles, rhs_tile, acc, config,
                listener,
            ),
            RhsTile::Double(_) => {
                panic!("Stage double buffering not yet supported for unit matmul")
            }
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        let shape = config.accumulator_shape();
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

        let (m_acc_shape, n_acc_shape) = stage_config.accumulator_shape();
        let num_acc_n = stage_config.tiling_dimensions(Ident::Rhs).tile_count_col() / n_acc_shape;
        let m_unit_offset = m_acc_shape * (UNIT_POS / num_acc_n);
        let n_unit_offset = n_acc_shape * (UNIT_POS % num_acc_n);

        let mut m_iter = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime![m_acc_shape] {
            let mut n_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime![n_acc_shape] {
                let accumulator = Self::Accumulator::get_at(acc, m_iter, n_iter);
                TMM::write_results(accumulator, &mut smem_slice, stage_config.to_tmm_config());
                Self::Writer::write::<G>(
                    out,
                    smem_slice.to_slice(),
                    m_unit_offset + m_iter,
                    n_unit_offset + n_iter,
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
        Self::Writer::new(tensor, x_offset, y_offset, batch_offset)
    }
}
