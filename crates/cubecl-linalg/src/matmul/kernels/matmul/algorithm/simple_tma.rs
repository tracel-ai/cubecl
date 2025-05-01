use super::base;
use core::marker::PhantomData;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    MatmulProblem, MatmulSelection,
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

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        let num_planes = selection.tile_count.m.div_ceil(selection.rows_per_plane);
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }
}
