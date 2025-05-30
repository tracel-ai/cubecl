use crate::matmul::components::MatmulProblem;
use crate::matmul::components::global::UnitWriter;
use crate::matmul::components::stage::PartitionBuffering;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulConfigInput;
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::kernels::matmul::StageInput;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::partitioned_stage_matmul::PartitionedStageMatmul;
use super::partitioned_stage_matmul::StagePartitioner;

type UnitMatmul<MP, TMM, RL, RR> = PartitionedStageMatmul<MP, TMM, RL, RR, UnitPartitioner>;

pub struct UnitPartitioner {}

#[cube]
impl StagePartitioner for UnitPartitioner {
    type Writer<EO: Numeric> = UnitWriter<EO>;

    fn init_writer<EO: Numeric>(
        tensor: VirtualTensor<EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Writer<EO> {
        UnitWriter::<EO>::new(tensor, x_offset, y_offset, batch_offset)
    }

    fn position() -> u32 {
        UNIT_POS
    }

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32) {
        config.num_planes() * config.plane_dim()
    }
}

pub struct UnitMatmulFamily<TMM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TMM, RF)>,
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for UnitMatmulFamily<TMM, RF> {
    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        UnitMatmul<MP, TMM::Matmul<MP>, RF::Reader<MP::ES, TL>, RF::Reader<MP::ES, TR>>;
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> MatmulConfigFactory for UnitMatmulFamily<TMM, RF> {
    type Input = StageInput;
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_units_needed = config.tiling_scheme().partitions_in_stage_mn();
        let num_units = config.plane_dim() * config.num_planes();

        if num_units != num_units_needed {
            return Err(Box::new(format!(
                "Error: Number of units {num_units} should be {num_units_needed}."
            )));
        }

        if config.partition_buffering() == PartitionBuffering::Double
            && config.tiling_scheme().tiles_in_partition_n() < 2
        {
            return Err(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            ));
        }

        TMM::check_config(&config.tile_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tile_config)
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
        let tile_config = TMM::make_config(
            tile_input, problem, line_sizes, cube_dim, cube_count, quantized,
        );

        CommonStageConfig::new(
            tile_config,
            stage_input.tiling_scheme,
            cube_dim.y,
            quantized,
            stage_input.partition_buffering,
            stage_input.num_stages,
        )
    }
}
