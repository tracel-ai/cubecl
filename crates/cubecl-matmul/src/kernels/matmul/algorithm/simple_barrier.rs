use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use super::{MatmulSelection, base, plane_matmul_selection};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    global::{load::AsyncFullLoadingStrategy, single_stage::barrier::SimpleBarrierMatmulFamily},
    stage::{FullReaderFamily, PlaneMatmulFamily},
    tile,
};

pub struct SimpleBarrierAlgorithm<TMM, L: AsyncFullLoadingStrategy> {
    pub _tmm: PhantomData<TMM>,
    pub _l: PhantomData<L>,
}

impl<TMM, L> base::Algorithm for SimpleBarrierAlgorithm<TMM, L>
where
    TMM: tile::TileMatmulFamily,
    L: AsyncFullLoadingStrategy,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleBarrierMatmulFamily<Self::StageMatmul, L, L>;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
        _args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elem_stage,
            elem_acc,
            Default::default(),
        )
    }
}
