use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_core::ir::Elem;

use crate::components::MatmulProblem;
use crate::components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul};
use crate::components::global::load::sync_buffer_cyclic;
use crate::components::global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily;
use crate::components::stage::{
    BufferReaderFamily, FullReaderFamily, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::tile;

use super::{
    MatmulSelection, MultiRowStrategy, PlaneMatmulSelectionOptions, base, plane_matmul_selection,
};

pub struct OrderedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedSelectionArgs {
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl<TMM> base::Algorithm for OrderedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily,
{
    type SelectionArgs = OrderedSelectionArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
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
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elem_stage,
            elem_acc,
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
