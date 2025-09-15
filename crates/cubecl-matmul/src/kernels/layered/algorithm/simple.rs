use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_runtime::MmaConfig;
use std::marker::PhantomData;

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        MultiRowStrategy, TilingScheme,
        batch::{
            CubeCountPlanSelection, GlobalOrderSelection, HypercubeSelection,
            PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul, SmAllocation,
        },
        global::{
            load::{SyncFullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FillReaderFamily, FullReaderFamily, PartitionBuffering,
            PlaneMatmulFamily, RowMajorTilingOrder,
        },
        tile::{
            TileMatmulFamily,
            loader::{Filled, Strided},
        },
    },
    kernels::layered::{
        Algorithm,
        selector::{PlaneMatmulSelectionOptions, plane_matmul_selection},
    },
};

/// Plane accelerated single stage matmul with configurable loaders (default to cyclic)
pub struct SimpleAlgorithm<
    TMM,
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

#[derive(Default, Debug, Clone)]
pub struct SimpleArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl<TMM, LL, RL> Algorithm for SimpleAlgorithm<TMM, LL, RL>
where
    TMM: TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled>,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type SelectionArgs = SimpleArgs;
    type TileMatmul = TMM;
    type StageMatmul =
        PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily, FillReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        if args.multi_rows {
            selection_multi_rows::<R, TMM>(client, problem, plane_dim, elems)
        } else {
            plane_matmul_selection::<TMM, R>(
                client,
                problem,
                plane_dim,
                elems,
                PlaneMatmulSelectionOptions {
                    partition_buffering: Some(PartitionBuffering::Single),
                    tiny_selection_enabled: true,
                    ..Default::default()
                },
            )
        }
    }
}

fn selection_multi_rows<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    elems: MatmulElems,
) -> Result<MatmulSelection, MatmulSetupError> {
    let supported = |m: u32, n: u32, k: u32| {
        client.properties().features.cmma.contains(&MmaConfig {
            a_type: elems.lhs_register,
            b_type: elems.rhs_register,
            cd_type: elems.acc_register,
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

        Ok(MatmulSelection::builder(tiling_scheme, plane_dim)
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_config(hypercube)
            .build())
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

        Ok(MatmulSelection::builder(tiling_scheme, plane_dim)
            .partition_buffering(PartitionBuffering::Single)
            .hypercube_config(hypercube)
            .build())
    } else {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elems,
            PlaneMatmulSelectionOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        )
    }
}
