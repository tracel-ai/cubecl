use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};

use crate::components::{
    MatmulPrecision,
    global::{GlobalConfig, read::SyncStrategy},
};

#[cube]
pub trait BarrierKind {
    fn level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct CubeCoop {}
#[derive(CubeType)]
pub struct CubeManual {}

#[cube]
impl BarrierKind for CubeCoop {
    fn level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}

#[cube]
impl BarrierKind for CubeManual {
    fn level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

/// Asynchronous barrier for TMA loads
pub struct AsyncBarrier<Kind: BarrierKind> {
    _ty: PhantomData<Kind>,
}

#[cube]
impl<Kind: BarrierKind> SyncStrategy for AsyncBarrier<Kind> {
    type Barrier = Barrier;

    fn create_barrier() -> Self::Barrier {
        Barrier::new(Kind::level())
    }

    fn sync<MP: MatmulPrecision, G: GlobalConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: G,
    ) {
        barrier.arrive_and_wait();
    }
}
