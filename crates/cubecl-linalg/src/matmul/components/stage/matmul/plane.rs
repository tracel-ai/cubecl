use crate::matmul::components::global::GlobalWriter;
use crate::matmul::components::global::{AccumulatorLoader, TilewiseWriter};
use crate::matmul::components::stage::matmul::base::{
    execute_double_buffer, execute_single_buffer,
};
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::shared::{RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{NoEvent, StageBuffering, StageEventListener};
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

use super::StageVectorization;
use super::shared::Accumulators;

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

        // let num_accumulators = tile_count.m * tile_count.n;
        // let num_planes = cube_dim.y;
        // // TODO move panic in check_config
        // let num_acc_per_primitive = if num_accumulators % num_planes != 0 {
        //     panic!(
        //         "For plane matmul, number of tile matmuls {} must be divisible by number of planes {}",
        //         num_accumulators, num_planes
        //     );
        // } else {
        //     num_accumulators / num_planes
        // };
        // let acc_n = tile_count.n;
        // // TODO move panic in check_config
        // let acc_m = if num_acc_per_primitive % acc_n != 0 {
        //     panic!(
        //         "For plane matmul, number of tile matmuls per primitive {} must be divisible by number of tiles in n {}",
        //         num_acc_per_primitive, acc_n
        //     );
        // } else {
        //     num_acc_per_primitive / acc_n
        // };

        CommonStageConfig::new(
            tmm_config, tiling, cube_dim.y, quantized, buffering, num_stages, acc_shape,
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
    RL: StageToTileReader<MP::ES>,
    RR: StageToTileReader<MP::ES>,
> {
    _phantom: PhantomData<(MP, TMM, RL, RR)>,
}

#[cube]
impl<MP, TMM, RL, RR> StageMatmul<MP> for PlaneMatmul<MP, TMM, RL, RR>
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
    type Writer = TilewiseWriter<MP::EO>;

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
            RhsTile::Single(rhs_fragment) => execute_single_buffer::<MP, TMM, RL, RR, SEL>(
                UNIT_POS_Y * acc.shape.0,
                0,
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                config,
                listener,
            ),
            RhsTile::Double(rhs_fragments) => execute_double_buffer::<MP, TMM, RL, RR, SEL>(
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
                TMM::write_results(accumulator, &mut smem_slice, stage_config.to_tmm_config());
                Self::Writer::write::<G>(
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
        Accumulators::<MP, TMM>::new(config.accumulator_shape(), config)
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
