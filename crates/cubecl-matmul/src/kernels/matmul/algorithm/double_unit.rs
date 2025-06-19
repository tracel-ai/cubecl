use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::{
    components::{
        MatmulProblem,
        batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
        global::{
            load::sync_buffer_cyclic, multi_stage::double_buffering::DoubleBufferingMatmulFamily,
        },
        stage::{BufferReaderFamily, RowMajorTilingOrder, UnitMatmulFamily},
        tile::register::RegisterMatmul,
    },
    kernels::matmul::Algorithm,
};

use super::{MatmulSelection, unit_matmul_selection};

pub struct DoubleUnitAlgorithm<Dispatch = batch::TransposedPartitioner> {
    pub _dispatch: PhantomData<Dispatch>,
}

impl<P> Algorithm for DoubleUnitAlgorithm<P>
where
    P: Partitioner,
{
    type SelectionArgs = ();
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
    type GlobalMatmul = DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
        _args: &Self::SelectionArgs,
    ) -> MatmulSelection {
        unit_matmul_selection(problem, plane_dim, true)
    }
}
