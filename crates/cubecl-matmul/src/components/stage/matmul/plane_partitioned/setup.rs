use crate::components::MatmulLineSizes;
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
use crate::components::stage::matmul::plane_partitioned::PlaneMatmul;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::{AccR, AccS, ComputeResources};
use crate::components::{LhsR, global::PartitionedStageFamily};
use crate::components::{LhsS, tile::io::Strided};
use crate::components::{MatrixPrecision, global::PartitionedStage};
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Plane Matmul family for any precision
pub struct PlaneMatmulFamily<
    TM: TileMatmulFamily,
    StageLhs: StageFamily,
    StageRhs: StageFamily,
    StageAcc: StageFamily,
> {
    _phantom: PhantomData<(TM, StageLhs, StageRhs, StageAcc)>,
}

impl<
    TM: TileMatmulFamily<OutTile = Strided>,
    StageLhs: StageFamily<TileKind = TM::LhsTile>,
    StageRhs: StageFamily<TileKind = TM::RhsTile>,
    StageAcc: StageFamily<TileKind = TM::AccTile>,
> StageMatmulFamily for PlaneMatmulFamily<TM, StageLhs, StageRhs, StageAcc>
{
    type LhsStage = StageLhs;
    type RhsStage = StageRhs;
    type AccStage = StageAcc;
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulPrecision,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = PlaneMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
        StageLhs::Stage<LhsS<MP>, TL>,
        StageRhs::Stage<RhsS<MP>, TR>,
        StageAcc::Stage<AccS<MP>, TA>,
        PartitionedStage<AccS<MP>>,
    >;

    type Config = PlanePartitionedStageConfig<TM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_global_readers: Option<MaxGlobalReaderPlanes>,
        ordered: bool,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config =
            TM::setup::<LhsR<MP>, RhsR<MP>, AccR<MP>, R>(client, problem, selection, line_sizes)?;

        let compute_resources =
            if let ComputeResources::Planes(planes) = TM::computation_resources()? {
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
            max_global_readers,
            compute_planes,
        )?;

        PlanePartitionedStageConfig::new(
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
