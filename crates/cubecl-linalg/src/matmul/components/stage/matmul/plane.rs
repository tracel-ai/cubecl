use crate::matmul::components::global::PlaneWriter;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::StageBuffering;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulConfigInput;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{Ident, MatmulProblem};
use crate::matmul::components::{
    InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::kernels::matmul::StageInput;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::partitioned_stage_matmul::{PartitionedStageMatmul, StagePartitioner};

type PlaneMatmul<MP, TMM, RL, RR> = PartitionedStageMatmul<MP, TMM, RL, RR, PlanePartitioner>;

pub struct PlanePartitioner {}

#[cube]
impl StagePartitioner for PlanePartitioner {
    type Writer<EO: Numeric> = PlaneWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO> {
        PlaneWriter::<EO>::new(tensor, x_offset, y_offset, batch_offset)
    }

    fn position() -> u32 {
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
    type LhsReader = LRF;
    type RhsReader = RRF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        PlaneMatmul<MP, TMM::Matmul<MP>, LRF::Reader<MP::ES, TL>, RRF::Reader<MP::ES, TR>>;
}

impl<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> MatmulConfigFactory
    for PlaneMatmulFamily<TMM, LRF, RRF>
{
    type Input = StageInput;
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_acc = config.tiling_dimensions(Ident::Out).tile_count();
        let partition_shape = config.tiling_scheme().tiles_per_partition;
        let acc_per_plane = partition_shape.num_elems();

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

        if config.buffering() == StageBuffering::Double && partition_shape.n < 2 {
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
        stage_input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let tile_input = TileMatmulConfigInput {
            vectorization: stage_input.stage_vectorization,
            tile_size: stage_input.tiling_scheme.tile_size,
        };
        let tmm_config = TMM::make_config(
            tile_input, problem, line_sizes, cube_dim, cube_count, quantized,
        );

        CommonStageConfig::new(
            tmm_config,
            stage_input.tiling_scheme,
            cube_dim.y,
            quantized,
            stage_input.stage_buffering,
            stage_input.num_stages,
        )
    }
}
