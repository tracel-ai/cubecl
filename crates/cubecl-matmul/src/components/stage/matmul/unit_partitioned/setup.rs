use crate::components::ComputeResources;
use crate::components::MatmulLineSizes;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaders;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::unit_partitioned::UnitMatmul;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
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
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_loaders: Option<MaxLoaders>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TMM::setup::<MP, R>(client, problem, selection, line_sizes)?;

        let compute_resources = if let ComputeResources::Units(units) =
            TMM::computation_resources()?
        {
            ComputeResources::Units(units * selection.tiling_scheme.stage_partitions_in_stage_mn())
        } else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Tried to use a unit stage matmul with a plane tile matmul.".to_string(),
            )));
        };

        let compute_planes = compute_resources.num_planes(tile_config.plane_dim())?;

        let plane_role_config = PlaneRoleConfig::new(
            selection.load_specialization_config,
            max_loaders,
            compute_planes,
        )?;

        UnitPartitionedStageConfig::new(
            tile_config,
            selection.tiling_scheme,
            selection.quantized,
            selection.partition_buffering,
            num_stages,
            plane_role_config,
            MP::ES::elem_size(),
            MP::EO::elem_size(),
            client.properties().hardware.max_shared_memory_size as u32,
            ordered,
        )
    }
}
