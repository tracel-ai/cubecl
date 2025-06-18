use core::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::{
    components::{
        MatmulLayouts, MatmulProblem,
        batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
        global::single_stage::tma::SimpleTmaMatmulFamily,
        stage::{FullReaderFamily, PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::matmul::Algorithm,
};

use super::{MatmulSelection, MultiRowStrategy, plane_matmul_selection};

pub struct SimpleTmaAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _tmm: PhantomData<TMM>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, P> Algorithm for SimpleTmaAlgorithm<TMM, P>
where
    TMM: TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleTmaMatmulFamily<Self::StageMatmul>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
        _layouts: MatmulLayouts,
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
