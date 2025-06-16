use std::marker::PhantomData;

use crate::{
    components::{
        batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
        global::{
            load::sync_buffer_cyclic, multi_stage::double_buffering::DoubleBufferingMatmulFamily,
        },
        stage::{BufferReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
        tile::register::RegisterMatmul,
    },
    kernels::matmul::Algorithm,
};

pub struct DoubleUnitAlgorithm<Dispatch = batch::TransposedPartitioner> {
    pub _dispatch: PhantomData<Dispatch>,
}

impl<P> Algorithm for DoubleUnitAlgorithm<P>
where
    P: Partitioner,
{
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;
}
