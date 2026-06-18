use cubecl_core::ir::{dialect::atomic::*, prelude::*};

use crate::hip::hip_op_with_out;

hip_op_with_out!(AtomicAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicAdd({ptr}, {value})")
});
