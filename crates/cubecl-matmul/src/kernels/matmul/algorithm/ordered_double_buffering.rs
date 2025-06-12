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
}
