use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierToken},
};

use crate::components::{
    LhsS, MatmulPrecision, RhsS,
    global::{GlobalConfig, SharedGlobalMatmulConfig, read::SyncStrategy},
    stage::StageConfig,
};

/// Asynchronous barrier for TMA loads
pub struct AsyncTma {}

#[cube]
impl SyncStrategy for AsyncTma {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        let bar = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        bar
    }

    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let num_bytes = comptime! {
            let lhs_bytes = config.lhs_reader_config().smem_config.elements_per_stage() * lhs_elem_size;
            let rhs_bytes = config.rhs_reader_config().smem_config.elements_per_stage() * rhs_elem_size;
            lhs_bytes + rhs_bytes
        };
        let token = arrive_tma(barrier, num_bytes);
        barrier.wait(token);
    }
}

#[cube]
/// Barrier for TMA
pub fn arrive_tma(barrier: &Barrier, #[comptime] num_bytes: u32) -> BarrierToken {
    let expected = select(UNIT_POS == 0, num_bytes, 0);
    barrier.arrive_and_expect_tx(1, expected)
}
