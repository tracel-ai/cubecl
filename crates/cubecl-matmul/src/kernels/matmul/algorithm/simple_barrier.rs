use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{load::AsyncFullLoadingStrategy, single_stage::barrier::SimpleBarrierMatmulFamily},
    stage::{FullReaderFamily, PlaneMatmulFamily},
    tile,
};

pub struct SimpleBarrierAlgorithm<
    TMM,
    L: AsyncFullLoadingStrategy,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _tmm: PhantomData<TMM>,
    pub _l: PhantomData<L>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, L, P> base::Algorithm for SimpleBarrierAlgorithm<TMM, L, P>
where
    TMM: tile::TileMatmulFamily,
    L: AsyncFullLoadingStrategy,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleBarrierMatmulFamily<Self::StageMatmul, L, L>;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}
