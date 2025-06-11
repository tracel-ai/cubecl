use crate::components::ComputeResources;
use crate::components::MatmulProblem;
use crate::components::TilingScheme;
use crate::components::global::PlaneRoleConfig;
use crate::components::global::PlaneWriter;
use crate::components::global::RoleRule;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::ReaderFamily;
use crate::components::stage::shared::CommonStageConfig;
use crate::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulConfigInput;
use crate::components::tile::TileMatmulFamily;
use crate::components::{
    InvalidConfigError, MatmulConfigFactory, MatmulLineSizes, MatmulPrecision,
};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::StageInput;
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

    fn position<S: StageConfig>(#[comptime] config: S) -> u32 {
        RoleRule::new(config.role_rule_config()).compute_index()
    }

    fn num_primitives<S: StageConfig>(#[comptime] config: S) -> comptime_type!(u32) {
        config.num_main_flow_planes()
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

    fn computation_resources(
        tiling_scheme: &TilingScheme,
    ) -> Result<ComputeResources, InvalidConfigError> {
        if let ComputeResources::Planes(planes) = TMM::computation_resources()? {
            Ok(ComputeResources::Planes(
                planes * tiling_scheme.stage_partitions_in_stage_mn(),
            ))
        } else {
            unreachable!("Plane matmul should not demand units")
        }
    }
}

impl<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> MatmulConfigFactory
    for PlaneMatmulFamily<TMM, LRF, RRF>
{
    type Input = StageInput;
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_planes_needed = config.tiling_scheme().stage_partitions_in_stage_mn();
        let num_compute_planes = config.num_main_flow_planes();

        if num_compute_planes != num_planes_needed {
            return Err(Box::new(format!(
                "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
            )));
        }

        // TODO we should allow buffering on m dimension
        if config.partition_buffering() == PartitionBuffering::Double
            && config.tiling_scheme().tiles_in_stage_partition_n() < 2
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

        let compute_planes =
            <Self as StageMatmulFamily>::computation_resources(&stage_input.tiling_scheme)
                .unwrap_or_else(|e| panic!("{}", e))
                .as_plane_resources(tile_config.plane_dim())
                .unwrap_or_else(|e| panic!("{}", e))
                .get_count();
        let plane_role_config = PlaneRoleConfig::from_plane_roles(
            stage_input
                .load_specialization
                .to_plane_roles(compute_planes),
        );

        CommonStageConfig::new(
            tile_config,
            stage_input.tiling_scheme,
            quantized,
            stage_input.partition_buffering,
            stage_input.num_stages,
            plane_role_config,
        )
    }
}
