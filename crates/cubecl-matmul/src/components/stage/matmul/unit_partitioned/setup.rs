use crate::components::AvailableLineSizes;
use crate::components::ComputeResources;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::global::LoaderTasksMap;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::unit_partitioned::UnitMatmul;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
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
    type Config = UnitPartitionedStageConfig<TMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
        num_stages: NumStages,
        loader_tasks_map: Option<LoaderTasksMap>,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TMM::setup::<MP, R>(client, problem, selection, available_line_sizes)?;

        let compute_resources = if let ComputeResources::Units(units) =
            TMM::computation_resources()?
        {
            ComputeResources::Units(units * selection.tiling_scheme.stage_partitions_in_stage_mn())
        } else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
            )));
        };

        let compute_planes = compute_resources
            .as_plane_resources(tile_config.plane_dim())?
            .get_count();

        let plane_role_config = PlaneRoleConfig::new(
            selection.load_specialization_config,
            loader_tasks_map,
            compute_planes,
            &tile_config,
        )?;

        UnitPartitionedStageConfig::new(
            tile_config,
            selection.tiling_scheme,
            selection.quantized,
            selection.partition_buffering,
            num_stages,
            plane_role_config,
        )
    }
}
