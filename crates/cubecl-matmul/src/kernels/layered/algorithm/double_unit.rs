use cubecl_core::{Runtime, client::ComputeClient};

use crate::{
    components::{
        MatmulElems, MatmulProblem, MatmulSelection,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            load::sync_partial_cyclic::SyncPartialCyclicLoading,
            multi_stage::double_buffering::DoubleBufferingMatmulFamily,
        },
        stage::{PartialReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
        tile::register::RegisterMatmul,
    },
    kernels::layered::{
        Algorithm,
        selector::{TileSizeSelection, UnitMatmulSelectionOptions, unit_matmul_selection},
    },
};

/// Unit double buffered matmul with cyclic loaders
pub struct DoubleUnitAlgorithm {}

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Algorithm for DoubleUnitAlgorithm {
    type SelectionArgs = DoubleUnitSelectionArgs;
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, PartialReaderFamily>;
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
        _elems: MatmulElems,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        unit_matmul_selection::<R>(
            client,
            problem,
            plane_dim,
            true,
            UnitMatmulSelectionOptions {
                tile: args.tile_size,
                ..Default::default()
            },
        )
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
