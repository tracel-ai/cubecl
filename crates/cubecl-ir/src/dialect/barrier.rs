use cubecl_macros_internal::{cube_op, op_traits};
use pliron::r#type::{TypeHandle, TypedHandle};

use crate::{
    CanMaterialize, HasSideEffects,
    attributes::{BoolAttr, IndexAttr},
    dialect::synchronization::SyncScope,
    interfaces::{
        Synchronizes,
        side_effects::{Extent, MemoryEffect, MemoryEffectsOp, MemoryLocation},
    },
    prelude::*,
    types::{PointerType, barrier::BarrierTokenType},
};

#[cube_op(name = "barrier.init")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct InitOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
    pub arrival_count: Value,
}

// Opaque so we can't know exact memory effects. Treat it as atomic read-update.
fn barrier_effects(ptr: Value) -> Vec<MemoryEffect> {
    vec![
        MemoryEffect::Read(MemoryLocation::exact_pointer(ptr)),
        MemoryEffect::Write(MemoryLocation::exact_pointer(ptr)),
        MemoryEffect::MakeAvailable(MemoryLocation::exact_pointer(ptr)),
        MemoryEffect::MakeVisible(MemoryLocation::exact_pointer(ptr)),
    ]
}

#[cube_op(name = "barrier.memcpy_async")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct MemCopyAsyncOp {
    pub barrier: Value,
    pub source: Value,
    pub destination: Value,
    pub source_length: Value,
    pub cooperative: BoolAttr,
}

#[op_interface_impl]
impl MemoryEffectsOp for MemCopyAsyncOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let mut effects = barrier_effects(self.barrier(ctx));
        effects.extend(vec![
            MemoryEffect::Read((self.source(ctx), Extent::Unknown).into()),
            MemoryEffect::Write((self.destination(ctx), Extent::Unknown).into()),
        ]);
        effects
    }
}

#[cube_op(name = "barrier.memcpy_async_tx")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct MemCopyAsyncTxOp {
    pub barrier: Value,
    pub source: Value,
    pub destination: Value,
    pub source_length: Value,
}

#[op_interface_impl]
impl MemoryEffectsOp for MemCopyAsyncTxOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let mut effects = barrier_effects(self.barrier(ctx));
        effects.extend(vec![
            MemoryEffect::Read((self.source(ctx), Extent::Unknown).into()),
            MemoryEffect::Write((self.destination(ctx), Extent::Unknown).into()),
        ]);
        effects
    }
}

#[cube_op(name = "barrier.copy_async")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct CopyAsyncOp {
    pub source: Value,
    pub destination: Value,
    pub source_length: Value,
    pub copy_length: IndexAttr,
    pub checked: BoolAttr,
}

#[op_interface_impl]
impl MemoryEffectsOp for CopyAsyncOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let length = self.copy_length(ctx).0;
        if self.checked(ctx).0 {
            vec![
                MemoryEffect::Read((self.source(ctx), Extent::AtMost(length)).into()),
                MemoryEffect::Write((self.destination(ctx), Extent::AtMost(length)).into()),
            ]
        } else {
            vec![
                MemoryEffect::Read((self.source(ctx), Extent::Exact(length)).into()),
                MemoryEffect::Write((self.destination(ctx), Extent::Exact(length)).into()),
            ]
        }
    }
}

#[cube_op(name = "barrier.arrive")]
#[result_ty(from_inputs = token_ty)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct ArriveOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
}

#[cube_op(name = "barrier.arrive_and_expect_tx")]
#[result_ty(from_inputs = |ctx, bar, _, _| token_ty(ctx, bar))]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct ArriveAndExpectTxOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
    pub arrive_count_update: Value,
    pub transaction_count_update: Value,
}

fn token_ty(ctx: &Context, barrier: &Value) -> TypeHandle {
    let bar_ptr = barrier.get_type(ctx).deref(ctx);
    let bar_ptr = bar_ptr.downcast_ref::<PointerType>().unwrap();
    let bar = TypedHandle::from_handle(bar_ptr.inner, ctx).expect("Should be barrier");
    BarrierTokenType::get(ctx, bar).into()
}

#[cube_op(name = "barrier.commit_copy_async")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct CommitCopyAsyncOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
}

#[cube_op(name = "barrier.expect_tx")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct ExpectTxOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
    pub transaction_count_update: Value,
}

#[cube_op(name = "barrier.wait")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct WaitOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
    pub token: Value,
}

/// Sync scope depends on init params and use, so can't be trivially analyzed. So just be
/// conservative.
#[op_interface_impl]
impl Synchronizes for WaitOp {
    fn minimum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Unit
    }

    fn maximum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Device
    }
}

#[cube_op(name = "barrier.wait_parity")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct WaitParityOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
    pub phase: Value,
}

/// Sync scope depends on init params and use, so can't be trivially analyzed. So just be
/// conservative.
#[op_interface_impl]
impl Synchronizes for WaitParityOp {
    fn minimum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Unit
    }

    fn maximum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Device
    }
}

#[cube_op(name = "barrier.arrive_and_wait")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct ArriveAndWaitOp {
    // Opaque so we can't know exact memory effects. Treat it as atomic read-update.
    #[operand(ptr_read, ptr_write, make_visible, make_available)]
    pub barrier: Value,
}

/// Sync scope depends on init params and use, so can't be trivially analyzed. So just be
/// conservative.
#[op_interface_impl]
impl Synchronizes for ArriveAndWaitOp {
    fn minimum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Unit
    }

    fn maximum_scope(&self, _ctx: &Context) -> SyncScope {
        SyncScope::Device
    }
}
