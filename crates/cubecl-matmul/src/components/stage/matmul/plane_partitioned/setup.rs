use crate::components::AvailableLineSizes;
use crate::components::ComputeResources;
use crate::components::MatmulChecker;
use crate::components::MatmulProblem;
use crate::components::TilingScheme;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::config::PartitionedStageConfig;
use crate::components::stage::matmul::plane_partitioned::PlaneMatmul;
use crate::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::tiling_scheme;
use crate::components::{InvalidConfigError, MatmulPrecision};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
use crate::kernels::matmul::StageInput;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

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

    type Input = StageInput;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
        num_stages: NumStages,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TMM::setup(problem, selection, available_line_sizes)?;

        let compute_resources =
            if let ComputeResources::Planes(planes) = TMM::computation_resources()? {
                ComputeResources::Planes(
                    planes * selection.tiling_scheme.stage_partitions_in_stage_mn(),
                )
            } else {
                panic!()
                // TODO Activate
                // return Err(Box::new(
                //     "Error: Tried to use a plane stage matmul with a unit tile matmul.".to_string(),
                // ));
            };

        let compute_planes = compute_resources
            .as_plane_resources(tile_config.plane_dim())?
            .get_count();
        let plane_role_config = PlaneRoleConfig::from_plane_roles(
            selection
                .load_specialization_config
                .to_plane_roles(compute_planes),
        );

        Ok(PartitionedStageConfig::new(
            tile_config,
            selection.tiling_scheme,
            selection.quantized,
            selection.partition_buffering,
            num_stages,
            plane_role_config,
        ))
    }
}

impl<TMM: TileMatmulFamily, LRF: ReaderFamily, RRF: ReaderFamily> MatmulChecker
    for PlaneMatmulFamily<TMM, LRF, RRF>
{
    type Config = PartitionedStageConfig<TMM::Config>;

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
}
