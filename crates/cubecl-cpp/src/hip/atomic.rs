use cubecl_core::ir::{dialect::atomic::*, prelude::*};

use crate::{hip::hip_op_with_out, shared::atomic::scoped_name};

hip_op_with_out!(AtomicIAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let add = scoped_name(ctx, op.get_operation(), "atomicAdd");
    format!("{add}({ptr}, {value})")
});

hip_op_with_out!(AtomicFAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let add = scoped_name(ctx, op.get_operation(), "atomicAdd");
    format!("{add}({ptr}, {value})")
});
