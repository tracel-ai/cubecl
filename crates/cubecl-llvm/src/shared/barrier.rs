//! Target-specific barrier lowering, and the token type an arrival returns.
//!
//! A cube barrier is an `mbarrier` on NVPTX, the one target that advertises barriers. On the
//! other targets the barrier operations lower to nothing. The copies that complete on a barrier
//! move data, so they have no such fallback, and only NVPTX takes them.

use crate::{
    prelude::*,
    shared::to_llvm::{erase, lower_by_target, nvptx_only},
};
use cubecl_core::ir::{
    dialect::barrier::{
        ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, ExpectTxOp, InitOp, MemCopyAsyncOp,
        MemCopyAsyncTxOp, WaitOp, WaitParityOp,
    },
    types::barrier::BarrierTokenType,
};

/// The barrier state an arrival returns, for the wait on its phase.
#[type_interface_impl]
impl CubeToLLVMType for BarrierTokenType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        IntegerType::get(ctx, 64, Signedness::Signless).into()
    }
}

/// Lowers `$cube_op` to an `mbarrier` operation on NVPTX, and drops it elsewhere.
macro_rules! barrier_op {
    ($cube_op:ty, $method:ident) => {
        lower_by_target!(
            $cube_op,
            ["nvptx" Nvptx => crate::nvptx::barrier::$method],
            erase
        );
    };
}

barrier_op!(InitOp, init);
barrier_op!(ArriveOp, arrive_op);
barrier_op!(ArriveAndExpectTxOp, arrive_and_expect_tx);
barrier_op!(ExpectTxOp, expect_tx);
barrier_op!(WaitOp, wait);
barrier_op!(WaitParityOp, wait_parity);
barrier_op!(ArriveAndWaitOp, arrive_and_wait);

nvptx_only!(MemCopyAsyncTxOp, barrier::memcpy_async_tx);
nvptx_only!(MemCopyAsyncOp, barrier::memcpy_async);
