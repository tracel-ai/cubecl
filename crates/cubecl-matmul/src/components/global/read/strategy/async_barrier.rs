use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};

use crate::components::{
    MatmulPrecision,
    global::{SharedGlobalMatmulConfig, read::SyncStrategy},
    stage::StageConfig,
};

#[cube]
pub trait BarrierKind {
    fn level() -> BarrierLevel;
}

/// Asynchronous barrier for `async_memcpy`
pub struct AsyncBarrier {}

#[cube]
impl SyncStrategy for AsyncBarrier {
    type Barrier = Barrier;

    fn create_barrier() -> Self::Barrier {
        Barrier::new(BarrierLevel::cube_full(UNIT_POS == 0))
    }

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.arrive_and_wait();
    }
}

/// Asynchronous barrier for `async_copy`
pub struct AsyncCopy {}

#[cube]
impl SyncStrategy for AsyncCopy {
    type Barrier = Barrier;

    fn create_barrier() -> Self::Barrier {
        Barrier::new(BarrierLevel::cube_full(UNIT_POS == 0))
    }

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.commit_copy_async();
        barrier.arrive_and_wait();
    }
}
