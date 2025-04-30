use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::matmul::components::MatmulProblem;
use crate::matmul::components::batch::{CubeCountDispatch, CubeDispatch};
use crate::matmul::components::global::load::SyncBufferLoadingStrategy;
use crate::matmul::components::stage::{
    self, BufferReaderFamily, 
};
use crate::matmul::components::{MatmulSelection, tile};
use crate::matmul::components::{batch, global};

use super::base;

pub struct DoubleBufferingAlgorithm<
    TMM,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    Dispatch = batch::TransposedDispatch,
> {
    pub _phantom: PhantomData<(TMM, LL, RL, Dispatch)>,
}

impl<TMM, LL, RL, Dispatch> base::Algorithm for DoubleBufferingAlgorithm<TMM, LL, RL, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<Self::TileMatmul, BufferReaderFamily>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        LL,
        RL,
    >;

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

    fn num_stages() -> u32 {
        2
    }
}
