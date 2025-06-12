use crate::components::AvailableLineSizes;
use crate::components::ComputeResources;
use crate::components::MatmulChecker;
use crate::components::MatmulProblem;
use crate::components::TilingScheme;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::PartitionedStageConfig;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::unit_partitioned::UnitMatmul;
use crate::components::stage::{StageConfig, StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::TileSetupInput;
use crate::components::{InvalidConfigError, MatmulPrecision};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
use crate::kernels::matmul::StageInput;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

pub struct UnitMatmulFamily<TMM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TMM, RF)>,
}

impl<TMM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for UnitMatmulFamily<TMM, RF> {
    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        UnitMatmul<MP, TMM::Matmul<MP>, RF::Reader<MP::ES, TL>, RF::Reader<MP::ES, TR>>;

    type Input = StageInput;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
        num_stages: NumStages,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TMM::setup(problem, selection, available_line_sizes)?;

        let compute_resources = if let ComputeResources::Units(units) =
            TMM::computation_resources()?
        {
            ComputeResources::Units(units * selection.tiling_scheme.stage_partitions_in_stage_mn())
        } else {
            panic!()
            // TODO Activate
            // return Err(Box::new(
            //     "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
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

impl<TMM: TileMatmulFamily, RF: ReaderFamily> MatmulChecker for UnitMatmulFamily<TMM, RF> {
    type Config = PartitionedStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let num_units_needed = config.tiling_scheme().stage_partitions_in_stage_mn();
        let num_units = config.plane_dim() * config.num_main_flow_planes();

        if num_units != num_units_needed {
            return Err(Box::new(format!(
                "Error: Number of units {num_units} should be {num_units_needed}."
            )));
        }

        if config.partition_buffering() == PartitionBuffering::Double
            && config.tiling_scheme().tiles_in_stage_partition_n() < 2
        {
            return Err(Box::new(
                "Error: Tried doing double buffering with only one tile to compute.".to_string(),
            ));
        }

        <TMM as MatmulChecker>::check_config(&config.tile_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tile_config)
    }
}
