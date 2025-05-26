use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::matmul::components::MatmulProblem;
use crate::matmul::components::batch::{CubeCountDispatch, CubeDispatch};
use crate::matmul::components::global::load::{sync_buffer_cyclic, sync_buffer_tilewise};
use crate::matmul::components::stage::{
    self, BufferReaderFamily, ColMajorTilingOrder, NumStages, RowMajorTilingOrder,
};
use crate::matmul::components::tile;
use crate::matmul::components::{batch, global};

use super::base::{self, MultiRowStrategy};
use super::{PlaneMatmulSelection, plane_matmul_selection};

pub struct CyclicDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

pub struct TilewiseDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

pub struct HybridDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

impl<TMM, Dispatch> base::Algorithm for CyclicDoubleBufferingAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = PlaneMatmulSelection;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_planes = selection.tile_count.m.div_ceil(selection.rows_per_plane);
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn num_stages() -> NumStages {
        (2, 2).into()
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

impl<TMM, Dispatch> base::Algorithm for TilewiseDoubleBufferingAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        // Other tiling orders are not supported
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_tilewise::LoadingStrategy<ColMajorTilingOrder>,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = PlaneMatmulSelection;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_planes = selection.tile_count.m.div_ceil(selection.rows_per_plane);
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn num_stages() -> NumStages {
        (2, 2).into()
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

impl<TMM, Dispatch> base::Algorithm for HybridDoubleBufferingAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch + CubeCountDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        BufferReaderFamily,
        BufferReaderFamily,
    >;
    type GlobalMatmul = global::multi_stage::double_buffering::DoubleBufferingMatmulFamily<
        Self::StageMatmul,
        sync_buffer_tilewise::LoadingStrategy<RowMajorTilingOrder>,
        sync_buffer_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    >;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;
    type MatmulSelection = PlaneMatmulSelection;

    fn cube_dim(selection: &Self::MatmulSelection) -> CubeDim {
        let num_planes = selection.tile_count.m.div_ceil(selection.rows_per_plane);
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &Self::MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn num_stages() -> NumStages {
        (2, 2).into()
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
