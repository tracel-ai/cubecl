use core::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{PlaneWriterFamily, single_stage::tma::SimpleTmaMatmulFamily},
        stage::{FilledStageFamily, PlaneMatmulFamily, StridedStageFamily},
        tile::{
            TileMatmulFamily,
            io::{Filled, Strided},
        },
    },
    kernels::layered::{Algorithm, selector::plane_matmul_selection},
};

/// Plane accelerated single stage matmul with tma loading
pub struct SimpleTmaAlgorithm<TMM> {
    pub _tmm: PhantomData<TMM>,
}

impl<TMM> Algorithm for SimpleTmaAlgorithm<TMM>
where
    TMM:
        TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled, OutTile = Strided>,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = SimpleTmaMatmulFamily<Self::StageMatmul, PlaneWriterFamily>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        elems: MatmulElems,
        _args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(client, problem, plane_dim, elems, Default::default())
    }
}
