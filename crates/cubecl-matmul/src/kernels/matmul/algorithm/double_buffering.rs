use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::MatmulProblem;
use crate::components::batch::{
    PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul,
};
use crate::components::global::load::{sync_buffer_cyclic, sync_buffer_tilewise};
use crate::components::stage::{
    BufferReaderFamily, ColMajorTilingOrder, NumStages, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::tile;
use crate::components::{batch, global};

use super::base::{self, MultiRowStrategy};
use super::{MatmulSelection, plane_matmul_selection};

pub struct CyclicDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

pub struct TilewiseDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

pub struct HybridDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

impl<TMM, P> base::Algorithm for CyclicDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, BufferReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}

impl<TMM, P> base::Algorithm for TilewiseDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, BufferReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        // Not sure if other tiling orders are supported
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_tilewise::LoadingStrategy<ColMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}

impl<TMM, P> base::Algorithm for HybridDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, BufferReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}
