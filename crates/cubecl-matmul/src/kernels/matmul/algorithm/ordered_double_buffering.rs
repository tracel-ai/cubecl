use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::batch::CubeDispatch;
use crate::components::global::load::sync_buffer_cyclic;
use crate::components::stage::{
    self, BufferReaderFamily, FullReaderFamily, NumStages, RowMajorTilingOrder,
};
use crate::components::tile;
use crate::components::{InvalidConfigError, MatmulProblem};
use crate::components::{batch, global};
use crate::kernels::matmul::Algorithm;

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

    fn cube_dim(selection: &MatmulSelection) -> Result<CubeDim, InvalidConfigError> {
        if selection.tiling_scheme.partitions_in_stage_n() > 1 {
            return Err(Box::new("Ordered does not support partitions > 1 in n"));
        }
        <Self as Algorithm>::cube_dim(selection)
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
