use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::global::{
    multi_stage::specialized::SpecializedMatmulFamily,
    read::async_partial_tma::AsyncPartialTmaLoading,
};
use crate::components::stage::FilledStageFamily;
use crate::components::stage::PlaneMatmulFamily;
use crate::components::{MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError};
use crate::components::{MatmulProblem, MultiRowStrategy, tile};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::components::{
    global::{
        PlaneWriterFamily,
        read::{PartialLoadingStrategy, async_tma::AsyncTma},
    },
    stage::StageFamily,
};
use crate::kernels::layered::algorithm::base;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

#[derive(Default, Debug, Clone, Copy)]
pub struct SpecializedArgs {
    pub swizzled: bool,
}

/// Plane accelerated specialized matmul with TMA readers
pub struct TmaSpecializedAlgorithm<TMM, L = AsyncPartialTmaLoading> {
    pub _phantom: PhantomData<(TMM, L)>,
}

impl<TMM, L> base::Algorithm for TmaSpecializedAlgorithm<TMM, L>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = <L::Stage as StageFamily>::TileKind,
            RhsTile = <L::Stage as StageFamily>::TileKind,
            AccTile = Filled,
            OutTile = Strided,
        >,
    L: PartialLoadingStrategy<SyncStrategy = AsyncTma>,
{
    type SelectionArgs = SpecializedArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, L::Stage, L::Stage, FilledStageFamily>;
    type GlobalMatmul = SpecializedMatmulFamily<Self::StageMatmul, L, L, PlaneWriterFamily>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        args: &Self::SelectionArgs,
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
                swizzled: args.swizzled,
                ..Default::default()
            },
        )
    }
}
