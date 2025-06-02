use super::{
    MatmulSelection, MultiRowStrategy, PlaneMatmulSelection, base, plane_matmul_selection,
};
use core::marker::PhantomData;
use cubecl_core::{ir::Elem, prelude::*};

use crate::components::{
    MatmulLineSizes, MatmulProblem,
    batch::{self, CubeCountDispatch, CubeDispatch},
    global::{self},
    stage::{self, FullReaderFamily},
    tile,
};

pub struct SimpleTmaAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _tmm: PhantomData<TMM>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, Dispatch> base::Algorithm for SimpleTmaAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        FullReaderFamily,
    >;
    type GlobalMatmul = global::single_stage::simple::SimpleTmaMatmulFamily<Self::StageMatmul>;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = PlaneMatmulSelection;

    fn line_sizes(
        problem: &MatmulProblem,
        _in_available: impl Iterator<Item = u8> + Clone,
        out_available: impl Iterator<Item = u8> + Clone,
        _selection: &Self::MatmulSelection,
    ) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: 1,
            rhs: 1,
            out: MatmulLineSizes::maximize_out(problem, out_available),
        }
    }

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_planes = selection.tiling_scheme().partitions_in_stage_m();
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
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> Self::MatmulSelection {
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
