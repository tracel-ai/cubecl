use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::MatmulProblem;
use crate::components::batch::CubeDispatch;
use crate::components::global::load::sync_buffer_cyclic;
use crate::components::stage::{
    self, BufferReaderFamily, FullReaderFamily, NumStages, RowMajorTilingOrder,
};
use crate::components::tile;
use crate::components::{batch, global};

use super::base::{self, MultiRowStrategy};
use super::{MatmulSelection, plane_matmul_selection};

pub struct OrderedDoubleBufferingAlgorithm<TMM, Dispatch = batch::TransposedDispatch> {
    pub _phantom: PhantomData<(TMM, Dispatch)>,
}

impl<TMM, Dispatch> base::Algorithm for OrderedDoubleBufferingAlgorithm<TMM, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    Dispatch: CubeDispatch,
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

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        let num_planes = selection.tiling_scheme.partitions_in_stage_m();
        CubeDim::new(selection.plane_dim, num_planes, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_in_stage_m();
        let n_stage = selection.tiling_scheme.elements_in_stage_n();
        let cubes_for_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_for_n = (problem.n as u32 + n_stage - 1) / n_stage;

        Dispatch::cube_count(cubes_for_m, cubes_for_n, problem.num_batches() as u32)
    }

    fn num_stages() -> NumStages {
        (1, 2).into()
    }

    fn partition_buffering_strategy() -> stage::PartitionBuffering {
        stage::PartitionBuffering::Single
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
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
