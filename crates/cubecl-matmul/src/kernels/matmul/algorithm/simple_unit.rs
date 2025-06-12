use super::{MatmulSelection, base, unit_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulLineSizes, MatmulProblem, MatrixLayout,
    batch::{self, PartitionedBatchMatmulFamily, Partitioner, RowMajorGlobalPartitionMatmul},
    global::{
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
        single_stage::simple::SimpleMatmulFamily,
    },
    stage::{
        ColMajorTilingOrder, FullReaderFamily, PartitionBuffering, RowMajorTilingOrder,
        UnitMatmulFamily,
    },
    tile::register::RegisterMatmul,
};

pub struct SimpleUnitAlgorithm<
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    Dispatch = batch::TransposedPartitioner,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<LL, RL, P> base::Algorithm for SimpleUnitAlgorithm<LL, RL, P>
where
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
    P: Partitioner,
{
    type TileMatmul = RegisterMatmul;
    type StageMatmul = UnitMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalMatmul = SimpleMatmulFamily<Self::StageMatmul, LL, RL>;

    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul, P>;

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

    fn partition_buffering_strategy() -> PartitionBuffering {
        PartitionBuffering::Single
    }
}
