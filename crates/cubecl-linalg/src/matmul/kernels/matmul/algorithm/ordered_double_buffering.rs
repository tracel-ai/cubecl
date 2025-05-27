use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::matmul::components::MatmulProblem;
use crate::matmul::components::batch::{CubeCountDispatch, CubeDispatch};
use crate::matmul::components::global::load::sync_buffer_cyclic;
use crate::matmul::components::stage::{
    self, BufferReaderFamily, FullReaderFamily, NumStages, RowMajorTilingOrder,
};
use crate::matmul::components::tile;
use crate::matmul::components::{batch, global};

use super::base::{self, MultiRowStrategy};
use super::{MatmulSelection, PlaneMatmulSelection, plane_matmul_selection};

pub struct OrderedDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

impl<TMM, Dispatch> base::Algorithm for OrderedDoubleBufferingAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::ordered::OrderedDoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = PlaneMatmulSelection;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_planes = selection.partitions_per_stage.m;
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count().m * selection.tile_shape.m;
        let n_stage = selection.tile_count().n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn num_stages() -> NumStages {
        (1, 2).into()
    }

    fn stage_buffering_strategy() -> stage::StageBuffering {
        stage::StageBuffering::Single
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
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}
