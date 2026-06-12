use cubecl_macros_internal::cube_op;

use crate::{
    attributes::{BoolAttr, IndexAttr},
    pliron::prelude::*,
    types::barrier::BarrierTokenType,
};

#[cube_op(name = "barrier.init")]
#[result_ty(none)]
pub struct InitOp {
    barrier: Value,
    arrival_count: Value,
}

#[cube_op(name = "barrier.memcpy_async")]
#[result_ty(none)]
pub struct MemCopyAsyncOp {
    barrier: Value,
    #[operand(ptr_read)]
    source: Value,
    #[operand(ptr_write)]
    destination: Value,
    source_length: Value,
    cooperative: BoolAttr,
}

#[cube_op(name = "barrier.memcpy_async_tx")]
#[result_ty(none)]
pub struct MemCopyAsyncTxOp {
    barrier: Value,
    #[operand(ptr_read)]
    source: Value,
    #[operand(ptr_write)]
    destination: Value,
    source_length: Value,
}

#[cube_op(name = "barrier.copy_async")]
#[result_ty(none)]
pub struct CopyAsyncOp {
    #[operand(ptr_read)]
    source: Value,
    #[operand(ptr_write)]
    destination: Value,
    source_length: Value,
    copy_length: IndexAttr,
    checked: BoolAttr,
}

#[cube_op(name = "barrier.arrive")]
#[result_ty(fixed = BarrierTokenType::get(ctx).into())]
pub struct ArriveOp {
    barrier: Value,
}

#[cube_op(name = "barrier.arrive_and_expect_tx")]
#[result_ty(fixed = BarrierTokenType::get(ctx).into())]
pub struct ArriveAndExpectTxOp {
    barrier: Value,
    arrive_count_update: Value,
    transaction_count_update: Value,
}

#[cube_op(name = "barrier.commit_copy_async")]
#[result_ty(none)]
pub struct CommitCopyAsyncOp {
    barrier: Value,
}

#[cube_op(name = "barrier.expect_tx")]
#[result_ty(none)]
pub struct ExpectTxOp {
    barrier: Value,
    transaction_count_update: Value,
}

#[cube_op(name = "barrier.wait")]
#[result_ty(none)]
pub struct WaitOp {
    barrier: Value,
    token: Value,
}

#[cube_op(name = "barrier.wait_parity")]
#[result_ty(none)]
pub struct WaitParityOp {
    barrier: Value,
    phase: Value,
}

#[cube_op(name = "barrier.arrive_and_wait")]
#[result_ty(none)]
pub struct ArriveAndWaitOp {
    barrier: Value,
}
