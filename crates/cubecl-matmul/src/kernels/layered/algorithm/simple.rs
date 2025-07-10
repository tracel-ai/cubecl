use cubecl_core::{Feature, Runtime, client::ComputeClient, ir::Elem};
use std::marker::PhantomData;

use crate::{
    components::{
        MatmulProblem, MatmulSelection, MultiRowStrategy, TilingScheme,
        batch::{
            CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection,
            PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul, SmAllocation,
        },
        global::{
            load::{SyncFullLoadingStrategy, sync_full_cyclic},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FullReaderFamily, PartitionBuffering, PlaneMatmulFamily,
            RowMajorTilingOrder,
        },
        tile::TileMatmulFamily,
    },
    kernels::layered::{
        Algorithm,
        selector::{PlaneMatmulSelectionOptions, plane_matmul_selection},
    },
};

#[derive(Default, Debug, Clone)]
pub struct SimpleArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

pub struct SimpleAlgorithm<
    TMM,
    LL = sync_full_cyclic::SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = sync_full_cyclic::SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

impl<TMM, LL, RL> Algorithm for SimpleAlgorithm<TMM, LL, RL>
where
    TMM: TileMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type SelectionArgs = SimpleArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        if args.multi_rows {
            let supported = |m: u8, n: u8, k: u8| {
                client.properties().feature_enabled(Feature::Cmma {
                    a: elem_stage,
                    b: elem_stage,
                    c: elem_acc,
                    m,
                    n,
                    k,
                })
            };
            let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
                Some(num_sms) => CubeCountPlanSelection::Sm {
                    num_sms,
                    sm_usage: SmAllocation::Exact,
                    cubes_first: true,
                },
                None => CubeCountPlanSelection::Flattened,
            };

            if supported(8, 32, 16) {
                // A lot of multi-rows balanced with a
                // tile size of (8, 32, 16)
                let tiling_scheme = TilingScheme::builder()
                    .with_tile_size((8, 32, 16).into())
                    .with_partition_size((4, 4, 2).into())
                    .with_stage_size((4, 1, 1).into())
                    .build()
                    .unwrap();

                let hypercube = HypercubeSelection::builder(&tiling_scheme)
                    .global_order(GlobalOrderSelection::SwizzleRow {
                        m: problem.m as u32,
                        w: 4,
                    })
                    .cube_count_plan(cube_count_plan)
                    .build();

                MatmulSelection::builder(tiling_scheme, plane_dim)
                    .partition_buffering(PartitionBuffering::Single)
                    .hypercube_config(hypercube)
                    .build()
            } else if supported(8, 8, 8) {
                let tiling_scheme = TilingScheme::builder()
                    .with_tile_size((8, 8, 8).into())
                    .with_partition_size((4, 8, 2).into())
                    .with_stage_size((4, 1, 1).into())
                    .build()
                    .unwrap();
                let hypercube = HypercubeSelection::builder(&tiling_scheme)
                    .global_order(GlobalOrderSelection::SwizzleRow {
                        m: problem.m as u32,
                        w: 4,
                    })
                    .cube_count_plan(cube_count_plan)
                    .build();

                MatmulSelection::builder(tiling_scheme, plane_dim)
                    .partition_buffering(PartitionBuffering::Single)
                    .hypercube_config(hypercube)
                    .build()
            } else {
                plane_matmul_selection::<TMM, R>(
                    client,
                    problem,
                    plane_dim,
                    elem_stage,
                    elem_acc,
                    PlaneMatmulSelectionOptions {
                        partition_buffering: Some(PartitionBuffering::Single),
                        multi_row_strategy: MultiRowStrategy::Always(2),
                        partition_k: Some(2),
                        ..Default::default()
                    },
                )
            }
        } else {
            plane_matmul_selection::<TMM, R>(
                client,
                problem,
                plane_dim,
                elem_stage,
                elem_acc,
                PlaneMatmulSelectionOptions {
                    partition_buffering: Some(PartitionBuffering::Single),
                    ..Default::default()
                },
            )
        }
    }
}
