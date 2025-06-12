use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::batch;
use crate::components::batch::{
    PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul,
};
use crate::components::global::GlobalMatmulFamily;
use crate::components::global::load::sync_buffer_cyclic;
use crate::components::global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily;
use crate::components::stage::{
    BufferReaderFamily, FullReaderFamily, NumStages, PlaneMatmulFamily, RowMajorTilingOrder,
};
use crate::components::tile;
use crate::components::{InvalidConfigError, MatmulProblem};

use super::base::{self, MultiRowStrategy};
use super::{MatmulSelection, plane_matmul_selection};

pub struct OrderedDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

impl<TMM, P> base::Algorithm for OrderedDoubleBufferingAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, BufferReaderFamily>;
    type GlobalMatmul = OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn cube_dim(selection: &MatmulSelection) -> Result<CubeDim, InvalidConfigError> {
        if selection.tiling_scheme.stage_partitions_in_stage_n() > 1 {
            return Err(Box::new("Ordered does not support partitions > 1 in n"));
        }
        Self::GlobalMatmul::cube_dim(selection, Self::load_specialization_config())
    }

    fn num_stages() -> NumStages {
        (1, 2).into()
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
