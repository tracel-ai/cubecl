use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::stage::{PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::{
    MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
    global::PlaneWriterFamily,
};
use crate::components::{MultiRowStrategy, tile};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    stage::{FilledStageFamily, StridedStageFamily},
};
use crate::components::{
    global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily, tile::io::Filled,
};
use crate::components::{
    global::read::sync_partial_cyclic::SyncPartialCyclicLoading, tile::io::Strided,
};
use crate::kernels::layered::Algorithm;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

/// Plane accelerated double buffered matmul ordered on Lhs with cyclic reader on Rhs
pub struct OrderedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Debug, Clone, Default)]
pub struct OrderedSelectionArgs {
    pub partition_k: Option<u32>,
    pub row_count: Option<u32>,
    pub rows_per_plane: Option<u32>,
}

impl<TMM> Algorithm for OrderedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = OrderedSelectionArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        PlaneWriterFamily,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::SelectionArgs,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneMatmulSelectionOptions {
                partition_k: args.partition_k,
                row_count: args.row_count,
                multi_row_strategy: args
                    .rows_per_plane
                    .map(MultiRowStrategy::Always)
                    .unwrap_or_else(|| MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    }),
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}
