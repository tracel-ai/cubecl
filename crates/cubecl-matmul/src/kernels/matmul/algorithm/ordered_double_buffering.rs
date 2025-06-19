use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_core::ir::Elem;

use crate::components::batch::{
    PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul,
};
use crate::components::global::load::sync_buffer_cyclic;
use crate::components::global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily;
use crate::components::stage::{
    BufferReaderFamily, FullReaderFamily, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::tile;
use crate::components::{MatmulProblem, batch};

use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};

pub struct OrderedDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

#[derive(Debug, Clone)]
pub struct OrderedDoubleBufferingSelectionArgs {
    pub pk: Option<u32>,
    pub multi_rows: Option<MultiRowStrategy>,
    pub occupancy_factor: Option<u32>,
}

impl Default for OrderedDoubleBufferingSelectionArgs {
    fn default() -> Self {
        Self {
            pk: None,
            multi_rows: None,
            occupancy_factor: None,
        }
    }
}

impl<TMM, P> base::Algorithm for OrderedDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type SelectionArgs = OrderedDoubleBufferingSelectionArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
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
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            args.multi_rows
                .as_ref()
                .map(|m| m.clone())
                .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                }),
            elem_stage,
            elem_acc,
            args.pk,
            args.occupancy_factor,
            false,
        )
    }
}
