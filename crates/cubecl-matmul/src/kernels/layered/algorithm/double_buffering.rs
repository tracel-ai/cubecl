use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::global::read::{
    async_partial_cyclic::AsyncPartialCyclicLoading,
    async_partial_strided::AsyncPartialStridedLoading, async_partial_tma::AsyncPartialTmaLoading,
    sync_partial_cyclic::SyncPartialCyclicLoading,
};
use crate::components::global::{
    PlaneWriterFamily, read::sync_partial_tilewise::SyncPartialTilewiseLoading,
};
use crate::components::stage::{ColMajorTilingOrder, PlaneMatmulFamily, RowMajorTilingOrder};
use crate::components::{MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError};
use crate::components::{MatmulProblem, MultiRowStrategy, tile};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::components::{
    global::multi_stage::double_buffering::DoubleBufferingMatmulFamily,
    stage::{FilledStageFamily, StridedStageFamily},
};
use crate::kernels::layered::Algorithm;
use crate::kernels::layered::algorithm::base;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

/// Plane accelerated double buffered matmul with cyclic readers
pub struct CyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncCyclicDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise readers
pub struct TilewiseDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with tilewise reader on Lhs and cyclic on Rhs
pub struct HybridDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with TMA readers
pub struct TmaDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

/// Plane accelerated double buffered matmul with cyclic readers
pub struct AsyncStridedDoubleBufferingAlgorithm<TMM> {
    pub _phantom: PhantomData<TMM>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DoubleBufferingArgs {
    pub specialized: bool,
}

impl<TMM> base::Algorithm for CyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

impl<TMM> base::Algorithm for AsyncCyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        AsyncPartialCyclicLoading<RowMajorTilingOrder>,
        AsyncPartialCyclicLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

impl<TMM> Algorithm for TilewiseDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        // Other tiling orders are not supported
        SyncPartialTilewiseLoading<RowMajorTilingOrder>,
        SyncPartialTilewiseLoading<ColMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

impl<TMM> base::Algorithm for HybridDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialTilewiseLoading<RowMajorTilingOrder>,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

impl<TMM> base::Algorithm for TmaDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        AsyncPartialTmaLoading,
        AsyncPartialTmaLoading,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

impl<TMM> base::Algorithm for AsyncStridedDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = Filled,
            OutTile = Strided,
        >,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        FilledStageFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        AsyncPartialStridedLoading,
        AsyncPartialStridedLoading,
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
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}
