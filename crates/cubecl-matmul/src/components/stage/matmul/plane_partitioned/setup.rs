use crate::components::ComputeResources;
use crate::components::MatmulLineSizes;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaderPlanes;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::ReaderFamily;
use crate::components::stage::matmul::plane_partitioned::PlaneMatmul;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::TileSetupInfo;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Plane Matmul family for any precision
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
    type Config = PlanePartitionedStageConfig<TMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_loaders: Option<MaxLoaderPlanes>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TMM::setup::<MP, R>(
            client,
            TileSetupInfo::from_matmul(problem, selection, line_sizes),
        )?;

        let compute_resources =
            if let ComputeResources::Planes(planes) = TMM::computation_resources()? {
                ComputeResources::Planes(
                    planes * selection.tiling_scheme.stage_partitions_in_stage_mn(),
                )
            } else {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Error: Tried to use a plane stage matmul with a unit tile matmul.".to_string(),
                )));
            };

        let compute_planes = compute_resources.num_planes(tile_config.plane_dim())?;

        let plane_role_config = PlaneRoleConfig::new(
            selection.load_specialization_config,
            max_loaders,
            compute_planes,
        )?;

        PlanePartitionedStageConfig::new(
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
