use crate::matmul::components::global::TilewiseWriter;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::StageBuffering;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulConfigInput;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
    MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::stage_matmul_impl::{ExecutePrimitive, StageMatmulImpl};
use super::{AccumulatorCount, NumStages, StageVectorization};

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
type PlaneMatmul<MP, TMM, RL, RR> = StageMatmulImpl<MP, TMM, RL, RR, PlaneExecutionPrimitive>;

pub struct PlaneExecutionPrimitive {}

#[cube]
impl ExecutePrimitive for PlaneExecutionPrimitive {
    type Writer<EO: Numeric> = TilewiseWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO> {
        TilewiseWriter::<EO>::new(tensor, x_offset, y_offset, batch_offset)
    }

    fn id() -> u32 {
        UNIT_POS_Y
    }

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32) {
        config.num_planes()
    }
}

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
        NumStages,
        AccumulatorCount,
    );
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_acc = config.tiling_dimensions(Ident::Out).tile_count();
        let acc_per_plane = config.accumulator_count().num_tiles();

        if num_acc % acc_per_plane != 0 {
            return Err(Box::new(format!(
                "Error: Number of accumulators {num_acc} should be divisible by number of accumulators per plane {acc_per_plane}."
            )));
        }

        let num_planes_needed = num_acc / acc_per_plane;
        let num_planes = config.num_planes();

        if num_planes != num_planes_needed {
            return Err(Box::new(
                "Error: Number of planes {num_planes} should be {num_planes_needed}.".to_string(),
            ));
        }

        if config.buffering() == StageBuffering::Double && config.accumulator_count().n < 2 {
            return Err(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
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
        (tiling, buffering, vectorization, num_stages, acc_count): Self::Input,
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
            tmm_config, tiling, cube_dim.y, quantized, buffering, num_stages, acc_count,
        )
    }
}
