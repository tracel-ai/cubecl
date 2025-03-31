use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::matmul::components::batch::{CubeCountDispatch, CubeDispatch};
use crate::matmul::components::global::multi_stage::MaximizeSliceLengthBufferLoading;
use crate::matmul::components::stage;
use crate::matmul::components::{GlobalBuffering, MatmulProblem};
use crate::matmul::components::{MatmulSelection, tile};
use crate::matmul::components::{batch, global};

use super::base;

pub struct DoubleBufferingBarrierAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _tmm: PhantomData<TMM>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, Dispatch> base::Algorithm for DoubleBufferingBarrierAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::single_buffer::SingleBufferMatmulFamily<Self::TileMatmul>;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingBarrierMatmulFamily<
        Self::StageMatmul,
        MaximizeSliceLengthBufferLoading,
        MaximizeSliceLengthBufferLoading,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        CubeDim::new(selection.plane_dim, selection.tile_count.m, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn global_buffering() -> GlobalBuffering {
        GlobalBuffering::Double
    }
}
