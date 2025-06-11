use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::MatmulProblem;
use crate::components::batch::{Partitioner, RowMajorGlobalPartitionMatmul};
use crate::components::global::load::SyncBufferLoadingStrategy;
use crate::components::stage::{self, BufferReaderFamily, NumStages};
use crate::components::tile::{self, PlaneTile};
use crate::components::{batch, global};

use super::base::{self, MultiRowStrategy};
use super::{MatmulSelection, plane_matmul_selection};

pub struct DoubleBufferingAlgorithm<
    TMM,
    LL: SyncBufferLoadingStrategy,
    LR: SyncBufferLoadingStrategy,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _phantom: PhantomData<(TMM, LL, LR, Dispatch)>,
}

impl<TMM, LL, LR, P> base::Algorithm for DoubleBufferingAlgorithm<TMM, LL, LR, P>
where
    TMM: tile::TileMatmulFamily<PrimitiveTile = PlaneTile>,
    LL: SyncBufferLoadingStrategy,
    LR: SyncBufferLoadingStrategy,
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
        LL,
        LR,
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
            MultiRowStrategy::Never,
            // MultiRowStrategy::Adaptive {
            //     minimum_stage_count: 8,
            // },
            elem_stage,
            elem_acc,
        )
    }
}
