use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel, BarrierToken},
};

use crate::components::{
    LhsS, MatmulPrecision, RhsS,
    global::{GlobalConfig, read::SyncStrategy},
};

/// Asynchronous barrier for TMA loads
pub struct AsyncTma {}

#[cube]
impl SyncStrategy for AsyncTma {
    type Barrier = Barrier;

    fn create_barrier() -> Self::Barrier {
        Barrier::new_with_async_proxy_fence(BarrierLevel::cube_full(UNIT_POS == 0))
    }

    fn sync<MP: MatmulPrecision, G: GlobalConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: G,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size();
        let rhs_elem_size = RhsS::<MP>::type_size();
        let num_bytes = comptime! {
            let lhs_bytes = config.lhs_reader_config().smem_config.elements_in_stage() * lhs_elem_size;
            let rhs_bytes = config.rhs_reader_config().smem_config.elements_in_stage() * rhs_elem_size;
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
