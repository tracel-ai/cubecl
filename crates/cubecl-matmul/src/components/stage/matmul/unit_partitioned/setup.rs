use crate::components::ComputeResources;
use crate::components::InputPrecision;
use crate::components::LhsS;
use crate::components::MatmulLineSizes;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::RhsS;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaderPlanes;
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

/// Unit Matmul family for any precision
pub struct UnitMatmulFamily<TM: TileMatmulFamily, RF: ReaderFamily> {
    _phantom: PhantomData<(TM, RF)>,
}

impl<TM: TileMatmulFamily, RF: ReaderFamily> StageMatmulFamily for UnitMatmulFamily<TM, RF> {
    type LhsReader = RF;
    type RhsReader = RF;
    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> = UnitMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            MP::EA,
        >,
        RF::Reader<LhsS<MP>, TL>,
        RF::Reader<RhsS<MP>, TR>,
    >;
    type Config = UnitPartitionedStageConfig<TM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_loaders: Option<MaxLoaderPlanes>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TM::setup::<MP, R>(client, problem, selection, line_sizes)?;

        let compute_resources = if let ComputeResources::Units(units) = TM::computation_resources()?
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
            LhsS::<MP>::elem_size(),
            RhsS::<MP>::elem_size(),
            MP::EO::elem_size(),
            client.properties().hardware.max_shared_memory_size as u32,
            ordered,
        )
    }
}
