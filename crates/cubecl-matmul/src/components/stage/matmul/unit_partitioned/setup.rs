use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::RhsR;
use crate::components::RhsS;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxGlobalReaderPlanes;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::StageFamily;
use crate::components::stage::matmul::unit_partitioned::UnitMatmul;
use crate::components::stage::matmul::unit_partitioned::UnitPartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::{AccR, InputPrecision};
use crate::components::{AccS, ComputeResources};
use crate::components::{LhsR, tile::io::Strided};
use crate::components::{LhsS, global::WriteStageFamily};
use crate::components::{MatmulLineSizes, global::PartitionedStage};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Unit Matmul family for any precision
pub struct UnitMatmulFamily<TM: TileMatmulFamily, StageIn: StageFamily, StageAcc: StageFamily> {
    _phantom: PhantomData<(TM, StageIn, StageAcc)>,
}

impl<
    TM: TileMatmulFamily<
            LhsTile = StageIn::TileKind,
            RhsTile = StageIn::TileKind,
            AccTile = StageAcc::TileKind,
            OutTile = Strided,
        >,
    StageIn: StageFamily,
    StageAcc: StageFamily,
> StageMatmulFamily for UnitMatmulFamily<TM, StageIn, StageAcc>
{
    type LhsStage = StageIn;
    type RhsStage = StageIn;
    type AccStage = StageAcc;
    type OutStage = WriteStageFamily;

    type Matmul<
        MP: MatmulPrecision,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = UnitMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as InputPrecision>::Register,
            <MP::Rhs as InputPrecision>::Register,
            <MP::Acc as InputPrecision>::Register,
        >,
        StageIn::Stage<LhsS<MP>, TL>,
        StageIn::Stage<RhsS<MP>, TR>,
        StageAcc::Stage<AccS<MP>, TA>,
        PartitionedStage<AccS<MP>>,
    >;

    type Config = UnitPartitionedStageConfig<TM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_global_readers: Option<MaxGlobalReaderPlanes>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config =
            TM::setup::<LhsR<MP>, RhsR<MP>, AccR<MP>, R>(client, problem, selection, line_sizes)?;

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
            max_global_readers,
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
            AccS::<MP>::elem_size(),
            client.properties().hardware.max_shared_memory_size as u32,
            ordered,
        )
    }
}
