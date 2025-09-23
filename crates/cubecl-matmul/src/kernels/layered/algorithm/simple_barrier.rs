use cubecl_core::{Runtime, client::ComputeClient};

use std::marker::PhantomData;

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            read::AsyncFullLoadingStrategy, single_stage::barrier::SimpleBarrierMatmulFamily,
        },
        stage::{FilledStageFamily, PlaneMatmulFamily, StridedStageFamily},
        tile::{
            self,
            reader::{Filled, Strided},
        },
    },
    kernels::layered::{Algorithm, selector::plane_matmul_selection},
};

/// Plane accelerated single stage matmul with async barrier loading
pub struct SimpleBarrierAlgorithm<TMM, L: AsyncFullLoadingStrategy> {
    pub _tmm: PhantomData<TMM>,
    pub _l: PhantomData<L>,
}

impl<TMM, L> Algorithm for SimpleBarrierAlgorithm<TMM, L>
where
    TMM: tile::TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled>,
    L: AsyncFullLoadingStrategy,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = SimpleBarrierMatmulFamily<Self::StageMatmul, L, L>;

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
