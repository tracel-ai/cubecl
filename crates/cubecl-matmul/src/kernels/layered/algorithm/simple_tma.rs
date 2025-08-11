use core::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        MatmulElems, MatmulProblem, MatmulSelection,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::single_stage::tma::SimpleTmaMatmulFamily,
        stage::{FullReaderFamily, PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::layered::{Algorithm, selector::plane_matmul_selection},
};

/// Plane accelerated single stage matmul with tma loading
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
        elems: MatmulElems,
        _args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        plane_matmul_selection::<TMM, R>(client, problem, plane_dim, elems, Default::default())
    }
}
