use super::{UnitMatmulSelection, base, unit_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::matmul::components::{
    MatmulProblem,
    batch::{self, CubeCountDispatch, CubeDispatch},
    global::{
        self,
        load::{SyncFullLoadingStrategy, sync_full_cyclic, sync_full_cyclic_checked},
    },
    stage::{self, ColMajorTilingOrder, FullReaderFamily, RowMajorTilingOrder},
    tile,
};

pub struct SimpleUnitAlgorithm<
    LL = sync_full_cyclic_checked::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic_checked::LoadingStrategy<RowMajorTilingOrder>,
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

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_tile_matmuls = selection.tile_count.m * selection.tile_count.n;
        let plane_dim = selection.plane_dim;
        let num_planes = num_tile_matmuls.div_ceil(plane_dim);
        CubeDim::new(plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
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
        unit_matmul_selection::<Self::TileMatmul, R>(
            client, problem, plane_dim, elem_stage, elem_acc,
        )
    }
}
