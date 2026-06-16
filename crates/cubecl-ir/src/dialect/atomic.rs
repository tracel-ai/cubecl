use cubecl_macros_internal::cube_op;

use crate::{dialect::ptr_value_ty, interfaces::rematerialize, pliron::prelude::*};

macro_rules! atomic_binop {
    ($name: literal, $ty: ident) => {
        #[cube_op(name = $name)]
        #[result_ty(same_as = value)]
        pub struct $ty {
            #[operand(ptr_read, ptr_write)]
            ptr: Value,
            value: Value,
        }
        rematerialize!($ty);
    };
}

atomic_binop!("atomic.exchange", AtomicExchangeOp);
atomic_binop!("atomic.add", AtomicAddOp);
atomic_binop!("atomic.sub", AtomicSubOp);
atomic_binop!("atomic.min", AtomicMinOp);
atomic_binop!("atomic.max", AtomicMaxOp);
atomic_binop!("atomic.and", AtomicAndOp);
atomic_binop!("atomic.or", AtomicOrOp);
atomic_binop!("atomic.xor", AtomicXorOp);

#[cube_op(name = "atomic.load")]
#[result_ty(from_inputs = ptr_value_ty)]
pub struct AtomicLoadOp {
    #[operand(ptr_read)]
    ptr: Value,
}

#[cube_op(name = "atomic.store")]
#[result_ty(none)]
pub struct AtomicStoreOp {
    #[operand(ptr_write)]
    ptr: Value,
    value: Value,
}

#[cube_op(name = "atomic.compare_exchange_weak")]
#[result_ty(same_as = value)]
pub struct AtomicCompareExchangeWeakOp {
    ptr: Value,
    cmp: Value,
    value: Value,
}
