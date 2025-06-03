use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{self, CubeDispatch},
    global::{self, load::AsyncFullLoadingStrategy},
    stage::{self, FullReaderFamily},
    tile,
};

pub struct SimpleBarrierAlgorithm<
    TMM,
    L: AsyncFullLoadingStrategy,
    Dispatch = batch::TransposedDispatch,
> {
    pub _tmm: PhantomData<TMM>,
    pub _l: PhantomData<L>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, L, Dispatch> base::Algorithm for SimpleBarrierAlgorithm<TMM, L, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    L: AsyncFullLoadingStrategy,
    Dispatch: CubeDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        FullReaderFamily,
    >;
    type GlobalMatmul =
        global::single_stage::simple::SimpleBarrierMatmulFamily<Self::StageMatmul, L, L>;

    type BatchMatmul = batch::one_to_one::OneToOneMatmulFamily<Self::GlobalMatmul, Dispatch>;

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
            MultiRowStrategy::Never,
            elem_stage,
            elem_acc,
        )
    }
}
