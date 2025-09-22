use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            multi_stage::double_buffering::DoubleBufferingMatmulFamily,
            read::sync_partial_cyclic::SyncPartialCyclicLoading,
        },
        stage::{
            FillStageReaderFamily, PartialStageReaderFamily, RowMajorTilingOrder, UnitMatmulFamily,
        },
        tile::{reader::Filled, register::RegisterMatmul},
    },
    kernels::layered::{
        Algorithm,
        selector::{TileSizeSelection, UnitMatmulSelectionOptions, unit_matmul_selection},
    },
};

/// Unit double buffered matmul with cyclic readers
pub struct DoubleUnitAlgorithm {}

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Algorithm for DoubleUnitAlgorithm {
    type SelectionArgs = DoubleUnitSelectionArgs;
    type TileMatmul = RegisterMatmul<Filled>;
    type StageMatmul =
        UnitMatmulFamily<Self::TileMatmul, PartialStageReaderFamily, FillStageReaderFamily>;
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
        _elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(unit_matmul_selection::<R>(
            client,
            problem,
            plane_dim,
            true,
            UnitMatmulSelectionOptions {
                tile: args.tile_size,
                ..Default::default()
            },
        ))
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
