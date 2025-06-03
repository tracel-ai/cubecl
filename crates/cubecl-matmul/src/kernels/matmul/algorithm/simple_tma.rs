use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};
use core::marker::PhantomData;
use cubecl_core::{ir::Elem, prelude::*};

use crate::components::{
    MatmulLineSizes, MatmulProblem,
    batch::{self, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{self},
    stage::{self, FullReaderFamily},
    tile,
};

pub struct SimpleTmaAlgorithm<TMM, Dispatch = batch::TransposedPartitioner> {
    pub _tmm: PhantomData<TMM>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, P> base::Algorithm for SimpleTmaAlgorithm<TMM, P>
where
    TMM: tile::TileMatmulFamily,
    P: Partitioner,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        FullReaderFamily,
    >;
    type GlobalMatmul = global::single_stage::simple::SimpleTmaMatmulFamily<Self::StageMatmul>;

    type BatchMatmul = batch::partitioned_batch_matmul::PartitionedBatchMatmulFamily<
        Self::GlobalMatmul,
        RowMajorGlobalPartitionMatmul,
        P,
    >;

    fn line_sizes(
        problem: &MatmulProblem,
        _in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
        _selection: &MatmulSelection,
    ) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: 1,
            rhs: 1,
            out: MatmulLineSizes::maximize_out(problem, out_available),
        }
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
            MultiRowStrategy::Never,
            elem_stage,
            elem_acc,
        )
    }
}
