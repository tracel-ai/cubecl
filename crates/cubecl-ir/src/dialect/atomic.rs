use cubecl_macros_internal::{cube_op, op_traits};

use crate::{
    CanMaterialize, HasSideEffects,
    dialect::{
        ptr_value_ty,
        synchronization::{SyncScope, SyncScopeAttr},
    },
    prelude::*,
};
use pliron::{context::Ptr, dict_key, operation::Operation};

dict_key!(
    /// Key of the scope an atomic synchronizes at, when it is not the device's.
    ATTR_KEY_ATOMIC_SCOPE, "atomic_scope"
);

/// The scope the atomic `op` synchronizes at: [`SyncScope::Device`] unless
/// [`set_atomic_scope`] set another.
///
/// A scope below the device's is a promise that only the units sharing it touch the memory, so
/// a backend may honor it or keep the device's. [`SyncScope::System`] is not optional: the atomic
/// also synchronizes with the host and with kernels on other devices that bind the memory.
pub fn atomic_scope(ctx: &Context, op: Ptr<Operation>) -> SyncScope {
    op.deref(ctx)
        .attributes
        .get::<SyncScopeAttr>(&ATTR_KEY_ATOMIC_SCOPE)
        .map(|attr| attr.0)
        .unwrap_or(SyncScope::Device)
}

/// Make the atomic `op` synchronize at `scope` (see [`atomic_scope`]).
pub fn set_atomic_scope(ctx: &mut Context, op: Ptr<Operation>, scope: SyncScope) {
    if scope == SyncScope::Device {
        return;
    }
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_ATOMIC_SCOPE.clone(), SyncScopeAttr(scope));
}

macro_rules! atomic_binop {
    ($name: literal, $ty: ident) => {
        #[cube_op(name = $name)]
        #[result_ty(same_as = value)]
        #[op_traits(CanMaterialize, HasSideEffects)]
        pub struct $ty {
            #[operand(ptr_read, ptr_write)]
            pub ptr: Value,
            pub value: Value,
        }
    };
}

atomic_binop!("atomic.exchange", AtomicExchangeOp);
atomic_binop!("atomic.i_add", AtomicIAddOp);
atomic_binop!("atomic.f_add", AtomicFAddOp);
atomic_binop!("atomic.i_sub", AtomicISubOp);
atomic_binop!("atomic.f_sub", AtomicFSubOp);
atomic_binop!("atomic.s_min", AtomicSMinOp);
atomic_binop!("atomic.u_min", AtomicUMinOp);
atomic_binop!("atomic.f_min", AtomicFMinOp);
atomic_binop!("atomic.s_max", AtomicSMaxOp);
atomic_binop!("atomic.u_max", AtomicUMaxOp);
atomic_binop!("atomic.f_max", AtomicFMaxOp);
atomic_binop!("atomic.and", AtomicAndOp);
atomic_binop!("atomic.or", AtomicOrOp);
atomic_binop!("atomic.xor", AtomicXorOp);

#[cube_op(name = "atomic.load")]
#[result_ty(from_inputs = ptr_value_ty)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct AtomicLoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}

#[cube_op(name = "atomic.store")]
#[result_ty(none)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct AtomicStoreOp {
    #[operand(ptr_write)]
    pub ptr: Value,
    pub value: Value,
}

#[cube_op(name = "atomic.compare_exchange_weak")]
#[result_ty(same_as = value)]
#[op_traits(CanMaterialize, HasSideEffects)]
pub struct AtomicCompareExchangeWeakOp {
    #[operand(ptr_read, ptr_write)]
    pub ptr: Value,
    pub cmp: Value,
    pub value: Value,
}
