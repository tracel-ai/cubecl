use crate::components::MatmulElems;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::MatrixPrecision;
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
use crate::components::tile::io::Strided;
use crate::components::{AccS, ComputeResources};
use crate::components::{LhsS, global::PartitionedStageFamily};
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
    type OutStage = PartitionedStageFamily;

    type Matmul<
        MP: MatmulPrecision,
        TL: TilingLayout,
        TR: TilingLayout,
        TA: TilingLayout,
        TO: TilingLayout,
    > = UnitMatmul<
        MP,
        TM::Matmul<
            <MP::Lhs as MatrixPrecision>::Register,
            <MP::Rhs as MatrixPrecision>::Register,
            <MP::Acc as MatrixPrecision>::Register,
        >,
        StageIn::Stage<LhsS<MP>, TL>,
        StageIn::Stage<RhsS<MP>, TR>,
        StageAcc::Stage<AccS<MP>, TA>,
        PartitionedStage<AccS<MP>>,
    >;

    type Config = UnitPartitionedStageConfig<TM::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_global_readers: Option<MaxGlobalReaderPlanes>,
        ordered: bool,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TM::setup::<R>(client, problem, selection, line_sizes, dtypes)?;

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
            dtypes.lhs_stage.size() as u32,
            dtypes.rhs_stage.size() as u32,
            dtypes.acc_stage.size() as u32,
            client.properties().hardware.max_shared_memory_size as u32,
            ordered,
        )
    }
}
