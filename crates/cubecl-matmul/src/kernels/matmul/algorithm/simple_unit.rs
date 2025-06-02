use super::{MatmulSelection, UnitMatmulSelection, base, unit_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulLineSizes, MatmulProblem, MatrixLayout,
    batch::{self, CubeCountDispatch, CubeDispatch},
    global::{
        self,
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
    },
    stage::{self, ColMajorTilingOrder, FullReaderFamily, PartitionBuffering, RowMajorTilingOrder},
    tile,
};

pub struct SimpleUnitAlgorithm<
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    Dispatch = batch::TransposedDispatch,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<LL, RL, Dispatch> base::Algorithm for SimpleUnitAlgorithm<LL, RL, Dispatch>
where
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = tile::register_matmul::RegisterMatmul;
    type StageMatmul = stage::unit_matmul::UnitMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalMatmul = global::single_stage::simple::SimpleMatmulFamily<Self::StageMatmul, LL, RL>;
    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = UnitMatmulSelection;

    fn line_sizes(
        problem: &MatmulProblem,
        in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
        selection: &Self::MatmulSelection,
    ) -> MatmulLineSizes {
        let max_lhs = match problem.lhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme().elements_in_tile_k(),
            MatrixLayout::ColMajor => selection.tiling_scheme().elements_in_tile_m(),
        };
        let max_rhs = match problem.rhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme().elements_in_tile_n(),
            MatrixLayout::ColMajor => selection.tiling_scheme().elements_in_tile_k(),
        };
        let max_out = selection.tiling_scheme().elements_in_tile_n();

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

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_units_needed = selection.tiling_scheme().partitions_in_stage_mn();
        let num_planes = num_units_needed.div_ceil(selection.plane_dim);

        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme().elements_in_stage_m();
        let n_stage = selection.tiling_scheme().elements_in_stage_n();
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
    ) -> Self::MatmulSelection {
        unit_matmul_selection(problem, plane_dim)
    }

    fn partition_buffering_strategy() -> PartitionBuffering {
        PartitionBuffering::Single
    }
}
