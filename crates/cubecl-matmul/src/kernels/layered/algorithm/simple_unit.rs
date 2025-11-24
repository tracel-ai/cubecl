use cubecl_core::{Runtime, client::ComputeClient};

use std::marker::PhantomData;

use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection, MatmulSetupError,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            UnitWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{
            ColMajorTilingOrder, FilledStageFamily, RowMajorTilingOrder, StridedStageFamily,
            UnitMatmulFamily,
        },
        tile::{TileMatmulFamily, io::Filled, register::RegisterMatmul},
    },
    kernels::layered::{
        TileSizeSelection,
        selector::{
            PartitionScaling, StageScaling, UnitMatmulSelectionOptions, unit_matmul_selection,
        },
    },
};

use super::Algorithm;

/// Unit single stage matmul with configurable readers (default to cyclic)
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
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
{
    type SelectionArgs = SimpleUnitSelectionArgs;
    type TileMatmul = RegisterMatmul<Filled>;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, StridedStageFamily, FilledStageFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL, UnitWriterFamily>;

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
        Ok(unit_matmul_selection(
            client,
            problem,
            plane_dim,
            false,
            line_sizes,
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
                swizzle: <RegisterMatmul as TileMatmulFamily>::should_swizzle(client),
            },
            dtypes,
        ))
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
