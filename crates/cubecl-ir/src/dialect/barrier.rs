use cubecl_macros_internal::cube_op;
use pliron::r#type::{TypeHandle, TypedHandle};

use crate::{
    attributes::{BoolAttr, IndexAttr},
    dialect::synchronization::SyncScope,
    interfaces::synchronizes,
    prelude::*,
    types::barrier::BarrierTokenType,
};

#[cube_op(name = "barrier.init")]
#[result_ty(none)]
pub struct InitOp {
    pub barrier: Value,
    pub arrival_count: Value,
}

#[cube_op(name = "barrier.memcpy_async")]
#[result_ty(none)]
pub struct MemCopyAsyncOp {
    pub barrier: Value,
    #[operand(ptr_read)]
    pub source: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub source_length: Value,
    pub cooperative: BoolAttr,
}

#[cube_op(name = "barrier.memcpy_async_tx")]
#[result_ty(none)]
pub struct MemCopyAsyncTxOp {
    pub barrier: Value,
    #[operand(ptr_read)]
    pub source: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub source_length: Value,
}

#[cube_op(name = "barrier.copy_async")]
#[result_ty(none)]
pub struct CopyAsyncOp {
    #[operand(ptr_read)]
    pub source: Value,
    #[operand(ptr_write)]
    pub destination: Value,
    pub source_length: Value,
    pub copy_length: IndexAttr,
    pub checked: BoolAttr,
}

#[cube_op(name = "barrier.arrive")]
#[result_ty(from_inputs = token_ty)]
pub struct ArriveOp {
    pub barrier: Value,
}

#[cube_op(name = "barrier.arrive_and_expect_tx")]
#[result_ty(from_inputs = |ctx, bar, _, _| token_ty(ctx, bar))]
pub struct ArriveAndExpectTxOp {
    pub barrier: Value,
    pub arrive_count_update: Value,
    pub transaction_count_update: Value,
}

fn token_ty(ctx: &Context, barrier: &Value) -> TypeHandle {
    let bar = TypedHandle::from_handle(barrier.get_type(ctx), ctx).expect("Should be barrier");
    BarrierTokenType::get(ctx, bar).into()
}

#[cube_op(name = "barrier.commit_copy_async")]
#[result_ty(none)]
pub struct CommitCopyAsyncOp {
    pub barrier: Value,
}

#[cube_op(name = "barrier.expect_tx")]
#[result_ty(none)]
pub struct ExpectTxOp {
    pub barrier: Value,
    pub transaction_count_update: Value,
}

#[cube_op(name = "barrier.wait")]
#[result_ty(none)]
pub struct WaitOp {
    pub barrier: Value,
    pub token: Value,
}
// Assume largest scope for now in lieu of detailed analysis. This is conservative but safe.
synchronizes!(WaitOp, SyncScope::Cube);

#[cube_op(name = "barrier.wait_parity")]
#[result_ty(none)]
pub struct WaitParityOp {
    pub barrier: Value,
    pub phase: Value,
}
// Assume largest scope for now in lieu of detailed analysis. This is conservative but safe.
synchronizes!(WaitParityOp, SyncScope::Cube);

#[cube_op(name = "barrier.arrive_and_wait")]
#[result_ty(none)]
pub struct ArriveAndWaitOp {
    pub barrier: Value,
}
// Assume largest scope for now in lieu of detailed analysis. This is conservative but safe.
synchronizes!(ArriveAndWaitOp, SyncScope::Cube);
