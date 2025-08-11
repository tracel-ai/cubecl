use cubecl_core::{Runtime, client::ComputeClient};

use std::marker::PhantomData;

use crate::{
    components::{
        MatmulElems, MatmulProblem, MatmulSelection,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            load::{SyncFullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{ColMajorTilingOrder, FullReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
        tile::register::RegisterMatmul,
    },
    kernels::layered::{
        TileSizeSelection,
        selector::{
            PartitionScaling, StageScaling, UnitMatmulSelectionOptions, unit_matmul_selection,
        },
    },
};

use super::Algorithm;

/// Unit single stage matmul with configurable loaders (default to cyclic)
pub struct SimpleUnitAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

#[derive(Default, Clone, Debug)]
pub struct SimpleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl<LL, RL> Algorithm for SimpleUnitAlgorithm<LL, RL>
where
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type SelectionArgs = SimpleUnitSelectionArgs;
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;

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
            false,
            UnitMatmulSelectionOptions {
                tile: args.tile_size,
                stage: match args.tile_size {
                    TileSizeSelection::MinTileSize => StageScaling::Enabled(2),
                    TileSizeSelection::MaxTileSize => StageScaling::Disabled,
                },
                partition: match args.tile_size {
                    TileSizeSelection::MinTileSize => PartitionScaling::Disabled,
                    TileSizeSelection::MaxTileSize => PartitionScaling::Enabled,
                },
            },
        )
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
