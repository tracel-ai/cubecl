use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        PartitionSize, TileSize, TilingScheme,
        batch::{
            CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection,
            PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul, SmAllocation,
        },
        global::{
            PlaneWriterFamily,
            multi_stage::double_buffering::DoubleBufferingMatmulFamily,
            read::{
                sync_full_cyclic::SyncFullCyclicLoading,
                sync_partial_cyclic::SyncPartialCyclicLoading,
            },
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, PartitionBuffering, PlaneMatmulFamily,
            RowMajorTilingOrder, StridedStageFamily,
        },
        tile::{io::Filled, plane_vec_mat_inner_product::PlaneVecMatInnerProduct},
    },
    kernels::layered::Algorithm,
};

pub struct SimpleVecMatAlgorithm {}

impl Algorithm for SimpleVecMatAlgorithm {
    type SelectionArgs = ();
    type TileMatmul = PlaneVecMatInnerProduct<Filled>;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = SimpleMatmulFamily<
        Self::StageMatmul,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
        PlaneWriterFamily,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _args: &Self::SelectionArgs,
        _dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat(
            client,
            problem,
            (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
            plane_dim,
        ))
    }
}

pub struct DoubleVecMatAlgorithm {}

impl Algorithm for DoubleVecMatAlgorithm {
    type SelectionArgs = ();
    type TileMatmul = PlaneVecMatInnerProduct<Filled>;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<ColMajorTilingOrder>,
        PlaneWriterFamily,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _args: &Self::SelectionArgs,
        _dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat(
            client,
            problem,
            (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
            plane_dim,
        ))
    }
}

fn selection_vecmat<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    tile_size: TileSize,
    plane_dim: u32,
) -> MatmulSelection {
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size(tile_size)
        .with_partition_size(PartitionSize::new(1, 1, 1))
        .with_stage_size((1, 1, 1).into())
        .build()
        .unwrap();
    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanSelection::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanSelection::FromProblem,
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 2,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .build()
}
