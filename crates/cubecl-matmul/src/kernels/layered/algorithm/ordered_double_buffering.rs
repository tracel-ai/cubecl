use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul};
use crate::components::global::load::sync_partial_cyclic::SyncPartialCyclicLoading;
use crate::components::global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily;
use crate::components::stage::{
    FullReaderFamily, PartialReaderFamily, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::{MatmulElems, MatmulProblem, MatmulSelection};
use crate::components::{MultiRowStrategy, tile};
use crate::kernels::layered::Algorithm;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

/// Plane accelerated double buffered matmul ordered on Lhs with cyclic loader on Rhs
pub struct OrderedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedSelectionArgs {
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl<TMM> Algorithm for OrderedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily,
{
    type SelectionArgs = OrderedSelectionArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, PartialReaderFamily>;
    type GlobalMatmul = OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elems,
            PlaneMatmulSelectionOptions {
                partition_k: args.partition_k,
                row_count: args.row_count,
                multi_row_strategy: args
                    .rows_per_plane
                    .map(MultiRowStrategy::Always)
                    .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    }),
                ..Default::default()
            },
        )
    }
}
