use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl,
    prelude::barrier::{Barrier, BarrierLevel},
};

use crate::components::{
    LhsS, MatmulIdent, MatmulPrecision, RhsS,
    global::{
        GlobalConfig,
        read::{SyncStrategy, arrive_tma},
    },
};

/// Asynchronous barrier for TMA loads
pub struct AsyncTma {}

#[cube]
impl SyncStrategy for AsyncTma {
    type Barrier = Barrier;

    fn create_barrier() -> Self::Barrier {
        Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32))
    }

    fn sync<MP: MatmulPrecision, G: GlobalConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: G,
    ) {
        let lhs_elem_size = LhsS::<MP>::elem_size();
        let rhs_elem_size = RhsS::<MP>::elem_size();
        let num_bytes = comptime! {
            let lhs_bytes = config.stage_memory_config(MatmulIdent::Lhs).elements_in_stage() * lhs_elem_size;
            let rhs_bytes = config.stage_memory_config(MatmulIdent::Rhs).elements_in_stage() * rhs_elem_size;
            lhs_bytes + rhs_bytes
        };
        arrive_tma(barrier, num_bytes);
        barrier.wait();
    }
}
