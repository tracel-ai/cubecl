use super::{MatmulSelection, MultiRowStrategy, base, plane_matmul_selection};
use cubecl_core::{ir::Elem, prelude::*};
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    batch::{self, CubeDispatch},
    global::{
        self,
        load::{SyncFullLoadingStrategy, sync_full_cyclic},
    },
    stage::{self, ColMajorTilingOrder, FullReaderFamily, RowMajorTilingOrder},
    tile,
};

pub struct SimpleAlgorithm<
    TMM,
    LL = sync_full_cyclic::LoadingStrategy<ColMajorTilingOrder>,
    RL = sync_full_cyclic::LoadingStrategy<RowMajorTilingOrder>,
    Dispatch = batch::TransposedDispatch,
> {
    pub _tmm: PhantomData<TMM>,
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _dispatch: PhantomData<Dispatch>,
}

impl<TMM, LL, RL, Dispatch> base::Algorithm for SimpleAlgorithm<TMM, LL, RL, Dispatch>
where
    TMM: tile::TileMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
    Dispatch: CubeDispatch,
{
    type TileMatmul = TMM;
    type StageMatmul = stage::plane_matmul::PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        FullReaderFamily,
    >;
    type GlobalMatmul = global::single_stage::simple::SimpleMatmulFamily<Self::StageMatmul, LL, RL>;
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
