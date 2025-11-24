use crate::components::AccS;
use crate::components::ComputeResources;
use crate::components::LhsS;
use crate::components::MatmulElems;
use crate::components::MatmulLineSizes;
use crate::components::MatmulPrecision;
use crate::components::MatmulProblem;
use crate::components::MatmulSelection;
use crate::components::MatrixLayout;
use crate::components::RhsS;
use crate::components::TilingScheme;
use crate::components::error::MatmulSetupError;
use crate::components::global::MatmulPlaneCounts;
use crate::components::global::MaxGlobalReaderPlanes;
use crate::components::global::PartitionedStageFamily;
use crate::components::global::PlaneRoleConfig;
use crate::components::stage::NumStages;
use crate::components::stage::PartitionBuffering;
use crate::components::stage::PartitionSchedulerScheme;
use crate::components::stage::StageFamily;
use crate::components::stage::StageMemoryConfig;
use crate::components::stage::matmul::partition::SharedPartitionMatmulConfig;
use crate::components::stage::matmul::partitioned_matmul::PartitionMatmulConfig;
use crate::components::stage::matmul::plane_partitioned::PlaneMatmul;
use crate::components::stage::matmul::plane_partitioned::PlanePartitionedStageConfig;
use crate::components::stage::{StageMatmulFamily, TilingLayout};
use crate::components::tile::TileConfig;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::io::Strided;
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

    type Config = PartitionMatmulConfig<TM::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        num_stages: NumStages,
        max_global_readers: Option<MaxGlobalReaderPlanes>,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = TM::setup(client, problem, selection, line_sizes, dtypes)?;

        let compute_resources =
            if let ComputeResources::Planes(planes) = TM::computation_resources()? {
                ComputeResources::Planes(
                    planes
                        * selection.tiling_scheme.partitions_per_stage_along_m()
                        * selection.tiling_scheme.partitions_per_stage_along_n(),
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

        let plane_counts = MatmulPlaneCounts::new(
            selection.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let lhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.lhs,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.k,
            tiles_per_partition_along_row: selection.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: selection.tiling_scheme.partition_size.k as u32,
            partitions_per_stage_along_row: selection.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: selection.tiling_scheme.stage_size.k as u32,
            line_size: line_sizes.lhs as u32,
            matrix_layout: problem.lhs_layout,
            swizzle: selection.shared_swizzle.lhs,
            num_stages: num_stages.lhs,
        };

        let rhs_smem_config = StageMemoryConfig {
            num_planes: plane_counts.rhs,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.k,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: selection.tiling_scheme.partition_size.k as u32,
            tiles_per_partition_along_col: selection.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: selection.tiling_scheme.stage_size.k as u32,
            partitions_per_stage_along_col: selection.tiling_scheme.stage_size.n as u32,
            line_size: line_sizes.rhs as u32,
            matrix_layout: problem.rhs_layout,
            swizzle: selection.shared_swizzle.rhs,
            num_stages: num_stages.rhs,
        };

        let out_smem_config = StageMemoryConfig {
            num_planes: plane_counts.out,
            elements_per_tile_along_row: selection.tiling_scheme.tile_size.m,
            elements_per_tile_along_col: selection.tiling_scheme.tile_size.n,
            tiles_per_partition_along_row: selection.tiling_scheme.partition_size.m as u32,
            tiles_per_partition_along_col: selection.tiling_scheme.partition_size.n as u32,
            partitions_per_stage_along_row: selection.tiling_scheme.stage_size.m as u32,
            partitions_per_stage_along_col: selection.tiling_scheme.stage_size.n as u32,
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            swizzle: selection.shared_swizzle.out,
            num_stages: 1,
        };

        let stage_config = PartitionMatmulConfig::Plane(
            PlanePartitionedStageConfig::from_shared_partition_config(
                SharedPartitionMatmulConfig::new(
                    tile_config,
                    selection.tiling_scheme.partition_size,
                    selection.partition_buffering,
                    plane_role_config,
                    selection.plane_dim,
                    selection.tiling_scheme.stage_size,
                    PartitionSchedulerScheme::Naive,
                    lhs_smem_config,
                    rhs_smem_config,
                    out_smem_config,
                ),
            ),
        );

        validate::<TM::Config>(
            stage_config,
            dtypes.lhs_stage.size() as u32,
            dtypes.rhs_stage.size() as u32,
            dtypes.acc_stage.size() as u32,
            client.properties().hardware.max_shared_memory_size as u32,
            selection.tiling_scheme,
            selection.partition_buffering,
            num_stages,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn validate<TC: TileConfig>(
    stage_config: PartitionMatmulConfig<TC>,
    lhs_s_size: u32,
    rhs_s_size: u32,
    eo_size: u32,
    smem_limit: u32,
    tiling_scheme: TilingScheme,
    partition_buffering: PartitionBuffering,
    num_stages: NumStages,
) -> Result<PartitionMatmulConfig<TC>, MatmulSetupError> {
    let num_planes_needed =
        tiling_scheme.partitions_per_stage_along_m() * tiling_scheme.partitions_per_stage_along_n();
    let num_compute_planes = stage_config.shared().plane_role_config.main_flow_count();

    if num_compute_planes != num_planes_needed {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Error: Number of compute planes {num_compute_planes} should be {num_planes_needed}."
        ))));
    }

    if partition_buffering == PartitionBuffering::Double
        && tiling_scheme.tiles_per_stage_partition_along_n() < 2
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Error: Tried doing double buffering with only one tile to compute.".to_string(),
        )));
    }

    let lhs_smem_size = tiling_scheme.elements_per_stage_along_m()
        * tiling_scheme.elements_per_stage_along_k()
        * num_stages.lhs;
    let rhs_smem_size = tiling_scheme.elements_per_stage_along_k()
        * tiling_scheme.elements_per_stage_along_n()
        * num_stages.rhs;
    let out_smem_size = tiling_scheme.tile_size.m * tiling_scheme.tile_size.n * num_compute_planes;
    let smem_total_size =
        lhs_s_size * lhs_smem_size + rhs_s_size * rhs_smem_size + eo_size * out_smem_size;

    if smem_total_size > smem_limit {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "This algorithm needs {smem_total_size:?} shared memory bytes but hardware limit is {smem_limit:?}. "
        ))));
    }

    Ok(stage_config)
}
