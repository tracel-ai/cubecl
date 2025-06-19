use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{load::AsyncFullLoadingStrategy, single_stage::barrier::SimpleBarrierMatmulFamily},
    stage::{FullReaderFamily, PlaneMatmulFamily},
    tile,
};

pub struct SimpleBarrierAlgorithm<
    TMM,
    L: AsyncFullLoadingStrategy,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _tmm: PhantomData<TMM>,
    pub _l: PhantomData<L>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, L, P> base::Algorithm for SimpleBarrierAlgorithm<TMM, L, P>
where
    TMM: tile::TileMatmulFamily,
    L: AsyncFullLoadingStrategy,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleBarrierMatmulFamily<Self::StageMatmul, L, L>;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}
