use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use super::{MatmulSelection, TileSizeSelection, base, unit_matmul_selection};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    global::{
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
        single_stage::simple::SimpleMatmulFamily,
    },
    stage::{ColMajorTilingOrder, FullReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
    tile::register::RegisterMatmul,
};

#[derive(Default, Clone, Debug)]
pub struct SimpleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

pub struct SimpleUnitAlgorithm<
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
}

impl<LL, RL> base::Algorithm for SimpleUnitAlgorithm<LL, RL>
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
        _elem_stage: Elem,
        _elem_acc: Elem,
        args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        unit_matmul_selection::<R>(client, problem, plane_dim, false, args.tile_size)
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
