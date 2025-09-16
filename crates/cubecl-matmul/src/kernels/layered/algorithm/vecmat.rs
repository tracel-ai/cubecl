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
            load::{
                sync_full_cyclic::SyncFullCyclicLoading,
                sync_partial_cyclic::SyncPartialCyclicLoading,
            },
            multi_stage::double_buffering::DoubleBufferingMatmulFamily,
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FillStageReaderFamily, FullStageReaderFamily,
            PartialStageReaderFamily, PartitionBuffering, PlaneMatmulFamily, RowMajorTilingOrder,
        },
        tile::{loader::Filled, plane_vec_mat_inner_product::PlaneVecMatInnerProduct},
    },
    kernels::layered::Algorithm,
};

pub struct SimpleVecMatAlgorithm {}

impl Algorithm for SimpleVecMatAlgorithm {
    type SelectionArgs = ();
    type TileMatmul = PlaneVecMatInnerProduct<Filled>;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        FullStageReaderFamily,
        FullStageReaderFamily,
        FillStageReaderFamily,
    >;
    type GlobalMatmul = SimpleMatmulFamily<
        Self::StageMatmul,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _elems: MatmulElems,
        _args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat::<R>(
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
        PartialStageReaderFamily,
        PartialStageReaderFamily,
        FillStageReaderFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<ColMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _elems: MatmulElems,
        _args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(selection_vecmat::<R>(
            client,
            problem,
            (1, line_sizes.out as u32, plane_dim * line_sizes.lhs as u32).into(),
            plane_dim,
        ))
    }
}

fn selection_vecmat<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
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
