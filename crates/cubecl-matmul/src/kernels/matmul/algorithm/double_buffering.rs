use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::MatmulProblem;
use crate::components::batch::{Partitioner, RowMajorGlobalPartitionMatmul};
use crate::components::global::load::{sync_buffer_cyclic, sync_buffer_tilewise};
use crate::components::stage::{
    self, BufferReaderFamily, ColMajorTilingOrder, NumStages, RowMajorTilingOrder,
};
use crate::components::tile::{self, PlaneTile};
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
    TMM: tile::TileMatmulFamily<PrimitiveTile = PlaneTile>,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::partitioned_batch_matmul::PartitionedBatchMatmulFamily<
        Self::GlobalMatmul,
        RowMajorGlobalPartitionMatmul,
        P,
    >;

    fn num_stages() -> NumStages {
        (2, 2).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}

impl<TMM, P> base::Algorithm for TilewiseDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily<PrimitiveTile = PlaneTile>,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        // Other tiling orders are not supported
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_tilewise::LoadingStrategy<ColMajorTilingOrder>,
    >;

    type BatchMatmul = batch::partitioned_batch_matmul::PartitionedBatchMatmulFamily<
        Self::GlobalMatmul,
        RowMajorGlobalPartitionMatmul,
        P,
    >;

    fn num_stages() -> NumStages {
        (2, 2).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        plane_matmul_selection::<Self::TileMatmul, R>(
            client,
            problem,
            plane_dim,
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}

impl<TMM, P> base::Algorithm for HybridDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily<PrimitiveTile = PlaneTile>,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::partitioned_batch_matmul::PartitionedBatchMatmulFamily<
        Self::GlobalMatmul,
        RowMajorGlobalPartitionMatmul,
        P,
    >;

    fn num_stages() -> NumStages {
        (2, 2).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        plane_matmul_selection::<Self::TileMatmul, R>(
            client,
            problem,
            plane_dim,
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}
