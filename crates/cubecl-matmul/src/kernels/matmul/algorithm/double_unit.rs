use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::{
    components::{
        MatmulProblem,
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            load::sync_buffer_cyclic, multi_stage::double_buffering::DoubleBufferingMatmulFamily,
        },
        stage::{BufferReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
        tile::register::RegisterMatmul,
    },
    kernels::matmul::Algorithm,
};

use super::{MatmulSelection, TileSizeSelection, unit_matmul_selection};

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

pub struct DoubleUnitAlgorithm {}

impl Algorithm for DoubleUnitAlgorithm {
    type SelectionArgs = DoubleUnitSelectionArgs;
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
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
        unit_matmul_selection::<R>(client, problem, plane_dim, true, args.tile_size)
    }

    fn select_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> u32 {
        client.properties().hardware.plane_size_min
    }
}
