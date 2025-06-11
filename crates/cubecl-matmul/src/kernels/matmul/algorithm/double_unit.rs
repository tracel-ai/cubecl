use super::{MatmulSelection, base, unit_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulLineSizes, MatmulProblem, MatrixLayout,
    batch::{self, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{self, load::SyncBufferLoadingStrategy},
    stage::{self, BufferReaderFamily, NumStages},
    tile,
};

pub struct DoubleUnitAlgorithm<
    LL: SyncBufferLoadingStrategy,
    LR: SyncBufferLoadingStrategy,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _dispatch: PhantomData<Dispatch>,
    pub _ll: PhantomData<LL>,
    pub _lr: PhantomData<LR>,
}

impl<LL, LR, P> base::Algorithm for DoubleUnitAlgorithm<LL, LR, P>
where
    LL: SyncBufferLoadingStrategy,
    LR: SyncBufferLoadingStrategy,
    P: Partitioner,
{
    type TileMatmul = tile::register_matmul::RegisterMatmul;
    type StageMatmul = stage::unit_matmul::UnitMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
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

    fn line_sizes(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
        selection: &MatmulSelection,
    ) -> MatmulLineSizes {
        let max_lhs = match problem.lhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_k(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_m(),
        };
        let max_rhs = match problem.rhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_n(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_k(),
        };
        let max_out = selection.tiling_scheme.elements_in_tile_n();

        MatmulLineSizes {
            lhs: MatmulLineSizes::maximize_lhs(
                problem,
                in_available
                    .clone()
                    .filter(|line_size| *line_size <= max_lhs as u8),
            ),
            rhs: MatmulLineSizes::maximize_rhs(
                problem,
                in_available.filter(|line_size| *line_size <= max_rhs as u8),
            ),
            out: MatmulLineSizes::maximize_out(
                problem,
                out_available.filter(|line_size| *line_size <= max_out as u8),
            ),
        }
    }

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
    ) -> MatmulSelection {
        unit_matmul_selection(problem, plane_dim)
    }
}
