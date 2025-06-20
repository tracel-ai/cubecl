use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use super::{MatmulSelection, base, plane_matmul_selection};
use std::marker::PhantomData;

use crate::components::{
    LoadSpecializationConfig, MatmulProblem, TilingScheme,
    batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
        single_stage::simple::SimpleMatmulFamily,
    },
    stage::{
        ColMajorTilingOrder, FullReaderFamily, PartitionBuffering, PlaneMatmulFamily,
        RowMajorTilingOrder,
    },
    tile::TileMatmulFamily,
};

#[derive(Default, Debug, Clone)]
pub struct SimpleArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

pub struct SimpleAlgorithm<
    TMM,
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, LL, RL, P> base::Algorithm for SimpleAlgorithm<TMM, LL, RL, P>
where
    TMM: TileMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
    P: Partitioner,
{
    type SelectionArgs = SimpleArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        if args.multi_rows {
            let tiling_scheme = TilingScheme::builder()
                .with_tile_size((8, 32, 16).into())
                .with_partition_size((4, 4, 2).into())
                .with_stage_size((4, 1, 1).into())
                .build()
                .unwrap();

            MatmulSelection::builder(tiling_scheme, plane_dim)
                .partition_buffering(PartitionBuffering::Single)
                .build()
        } else {
            plane_matmul_selection::<TMM, R>(
                client,
                problem,
                plane_dim,
                elem_stage,
                elem_acc,
                super::PlaneMatmulSelectionOptions {
                    partition_buffering: Some(PartitionBuffering::Single),
                    ..Default::default()
                },
            )
        }
    }
}
