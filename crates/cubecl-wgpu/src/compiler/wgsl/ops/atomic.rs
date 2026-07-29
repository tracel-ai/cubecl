use cubecl_ir::dialect::atomic::*;

use crate::compiler::wgsl::{
    to_wgsl::{wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

wgsl_op_with_out!(AtomicExchangeOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicExchange({ptr}, {value})")
});

wgsl_op_with_out!(AtomicIAddOp, AtomicFAddOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicAdd({ptr}, {value})")
});
wgsl_op_with_out!(AtomicISubOp, AtomicFSubOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicSub({ptr}, {value})")
});

wgsl_op_with_out!(AtomicSMaxOp, AtomicUMaxOp, AtomicFMaxOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicMax({ptr}, {value})")
});
wgsl_op_with_out!(AtomicSMinOp, AtomicUMinOp, AtomicFMinOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicMin({ptr}, {value})")
});

wgsl_op_with_out!(AtomicAndOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicAnd({ptr}, {value})")
});
wgsl_op_with_out!(AtomicOrOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicOr({ptr}, {value})")
});
wgsl_op_with_out!(AtomicXorOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicXor({ptr}, {value})")
});

wgsl_op_with_out!(AtomicLoadOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    format!("atomicLoad({ptr})")
});
wgsl_op!(AtomicStoreOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicStore({ptr}, {value});\n")
});

wgsl_op_with_out!(AtomicCompareExchangeWeakOp; |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let cmp = op.cmp(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicCompareExchangeWeak({ptr}, {cmp}, {value}).old_value")
});
