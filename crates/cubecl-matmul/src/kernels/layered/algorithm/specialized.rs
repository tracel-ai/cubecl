use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::global::PlaneWriterFamily;
use crate::components::global::{
    multi_stage::specialized::SpecializedMatmulFamily,
    read::async_partial_tma::AsyncPartialTmaLoading,
};
use crate::components::stage::PlaneMatmulFamily;
use crate::components::stage::{FilledStageFamily, StridedStageFamily};
use crate::components::{MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError};
use crate::components::{MatmulProblem, MultiRowStrategy, tile};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::kernels::layered::algorithm::base;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

/// Plane accelerated specialized matmul with TMA readers
pub struct TmaSpecializedAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

impl<TMM> base::Algorithm for TmaSpecializedAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = SpecializedMatmulFamily<
        Self::StageMatmul,
        AsyncPartialTmaLoading,
        AsyncPartialTmaLoading,
        PlaneWriterFamily,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        _args: &Self::SelectionArgs,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            PlaneMatmulSelectionOptions {
                specialized: true,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                ..Default::default()
            },
        )
    }
}
