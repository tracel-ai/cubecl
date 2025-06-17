use core::marker::PhantomData;

use crate::{
    components::{
        batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
        global::single_stage::tma::SimpleTmaMatmulFamily,
        stage::{FullReaderFamily, PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::matmul::Algorithm,
};

pub struct SimpleTmaAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _tmm: PhantomData<TMM>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, P> Algorithm for SimpleTmaAlgorithm<TMM, P>
where
    TMM: TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalMatmul = SimpleTmaMatmulFamily<Self::StageMatmul>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}
