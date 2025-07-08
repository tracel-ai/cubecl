use core::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::{
    components::{
        MatmulProblem,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::single_stage::tma::SimpleTmaMatmulFamily,
        stage::{FullReaderFamily, PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::layered::{
        Algorithm,
        selector::{MatmulSelection, plane_matmul_selection},
    },
};

pub struct SimpleTmaAlgorithm<TMM> {
    pub _tmm: PhantomData<TMM>,
}

impl<TMM> Algorithm for SimpleTmaAlgorithm<TMM>
where
    TMM: TileMatmulFamily,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleTmaMatmulFamily<Self::StageMatmul>;
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
