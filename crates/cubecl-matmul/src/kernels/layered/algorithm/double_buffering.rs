use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::global::multi_stage::double_buffering::DoubleBufferingMatmulFamily;
use crate::components::global::read::sync_partial_tilewise::SyncPartialTilewiseLoading;
use crate::components::stage::{
    ColMajorTilingOrder, PartialStageReaderFamily, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::{MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError};
use crate::components::{MatmulProblem, MultiRowStrategy, tile};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::reader::{Filled, Strided},
};
use crate::components::{
    global::read::sync_partial_cyclic::SyncPartialCyclicLoading, stage::FillStageReaderFamily,
};
use crate::kernels::layered::Algorithm;
use crate::kernels::layered::algorithm::base;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};

/// Plane accelerated double buffered matmul with cyclic readers
pub struct CyclicDoubleBufferingAlgorithm<TMM> {
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

#[derive(Default, Debug, Clone, Copy)]
pub struct DoubleBufferingArgs {
    pub specialized: bool,
}

impl<TMM> base::Algorithm for CyclicDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled>,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        PartialStageReaderFamily,
        PartialStageReaderFamily,
        FillStageReaderFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elems,
            PlaneMatmulSelectionOptions {
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                ..Default::default()
            },
        )
    }
}

impl<TMM> Algorithm for TilewiseDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled>,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        PartialStageReaderFamily,
        PartialStageReaderFamily,
        FillStageReaderFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        // Other tiling orders are not supported
        SyncPartialTilewiseLoading<RowMajorTilingOrder>,
        SyncPartialTilewiseLoading<ColMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elems,
            PlaneMatmulSelectionOptions {
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                ..Default::default()
            },
        )
    }
}

impl<TMM> base::Algorithm for HybridDoubleBufferingAlgorithm<TMM>
where
    TMM: tile::TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = Filled>,
{
    type SelectionArgs = DoubleBufferingArgs;
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        PartialStageReaderFamily,
        PartialStageReaderFamily,
        FillStageReaderFamily,
    >;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        SyncPartialTilewiseLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            elems,
            PlaneMatmulSelectionOptions {
                specialized: args.specialized,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                ..Default::default()
            },
        )
    }
}
