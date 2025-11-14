use crate::components::MatmulElems;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::MatrixLayout;
use crate::components::MatrixPrecision;
use crate::components::RhsS;
use crate::components::TilingScheme;
use crate::components::error::MatmulSetupError;
use crate::components::global::MaxGlobalReaderPlanes;
use crate::components::global::PlaneRoleConfig;
use crate::components::global::TodoGRC;
use crate::components::stage::NumStages;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::PartitionSchedulerScheme;
use crate::components::stage::StageFamily;
use crate::components::stage::matmul::partition::SharedPartitionMatmulConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionMatmulConfig;
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

    type Config = PartitionMatmulConfig<TM::Config>;

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

        let tiling_scheme = selection.tiling_scheme;

        let execution_is_sync = {
            #[cfg(target_os = "macos")]
            {
                false
            }
            #[cfg(not(target_os = "macos"))]
            {
                true
            }
        };
        let must_sync_plane_after_execution = !execution_is_sync && ordered;

        let stage_config =
            PartitionMatmulConfig::Unit(UnitPartitionedStageConfig::from_shared_partition_config(
                SharedPartitionMatmulConfig::new(
                    tile_config,
                    tiling_scheme.partition_size,
                    must_sync_plane_after_execution,
                    TodoGRC,
                    TodoGRC,
                    selection.partition_buffering,
                    problem.lhs_layout,
                    problem.rhs_layout,
                    MatrixLayout::RowMajor,
                    plane_role_config.rule,
                    selection.plane_dim,
                    tiling_scheme.stage_size,
                    PartitionSchedulerScheme::Naive,
                ),
            ));

        validate::<TM::Config>(
            stage_config,
            dtypes.lhs_stage.size() as u32,
            dtypes.rhs_stage.size() as u32,
            dtypes.acc_stage.size() as u32,
            client.properties().hardware.max_shared_memory_size as u32,
            tiling_scheme,
            plane_role_config.main_flow_count(),
            selection.partition_buffering,
            selection.plane_dim,
            num_stages,
        )
    }
}

fn validate<TC: TileConfig>(
    stage_config: PartitionMatmulConfig<TC>,
    lhs_s_size: u32,
    rhs_s_size: u32,
    eo_size: u32,
    smem_limit: u32,
    tiling_scheme: TilingScheme,
    num_compute_planes: u32,
    partition_buffering: PartitionBuffering,
    plane_dim: u32,
    num_stages: NumStages,
) -> Result<PartitionMatmulConfig<TC>, MatmulSetupError> {
    let num_units_needed = tiling_scheme.stage_partitions_in_stage_mn();
    let num_units = plane_dim * num_compute_planes;

    if num_units != num_units_needed {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Error: Number of units {num_units} should be {num_units_needed}."
        ))));
    }

    if partition_buffering == PartitionBuffering::Double
        && tiling_scheme.tiles_in_stage_partition_n() < 2
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Error: Tried doing partition double buffering with only one tile to compute.",
        )));
    }

    let lhs_smem_size = tiling_scheme.elements_in_stage_mk() * num_stages.lhs;
    let rhs_smem_size = tiling_scheme.elements_in_stage_nk() * num_stages.rhs;
    let out_smem_size = tiling_scheme.elements_in_tile_mn() * num_units;
    let smem_total_size =
        lhs_s_size * lhs_smem_size + rhs_s_size * rhs_smem_size + eo_size * out_smem_size;

    if smem_total_size > smem_limit {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "This algorithm needs {smem_total_size:?} shared memory bytes but hardware limit is {smem_limit:?}. "
        ))));
    }

    Ok(stage_config)
}
