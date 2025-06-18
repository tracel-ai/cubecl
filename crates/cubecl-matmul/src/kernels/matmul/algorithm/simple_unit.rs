use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use super::{MatmulSelection, base, unit_matmul_selection};
use std::marker::PhantomData;

use crate::components::{
    MatmulLayouts, MatmulProblem,
    batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
        single_stage::simple::SimpleMatmulFamily,
    },
    stage::{ColMajorTilingOrder, FullReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
    tile::register::RegisterMatmul,
};

pub struct SimpleUnitAlgorithm<
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<LL, RL, P> base::Algorithm for SimpleUnitAlgorithm<LL, RL, P>
where
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
    P: Partitioner,
{
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
        layouts: MatmulLayouts,
    ) -> MatmulSelection {
        unit_matmul_selection(problem, layouts, plane_dim, false)
    }
}
