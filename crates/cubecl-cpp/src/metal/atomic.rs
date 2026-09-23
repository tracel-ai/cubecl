use cubecl_core::ir::{dialect::atomic::*, prelude::*};

use crate::{
    metal::{metal_op, metal_op_with_out},
    shared::{CppValue, scoped_block, ty::TypeExtCPP},
};

metal_op_with_out!(AtomicLoadOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    format!("atomic_load_explicit({ptr}, memory_order_relaxed)")
});

metal_op!(AtomicStoreOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_store_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicExchangeOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_exchange_explicit({ptr}, {value}, memory_order_relaxed)")
});

// MSL's compare-exchange is weak and answers with a `bool`, writing the value it saw into
// `expected` on failure; the op returns the value it saw, as CUDA's `atomicCAS` does. A failure
// that saw `cmp` is spurious and is retried, so the value returned is never a spurious miss.
metal_op_with_out!(AtomicCompareExchangeWeakOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let cmp = op.cmp(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let ty = op.cmp(ctx).get_type(ctx).to_cpp(ctx);
    scoped_block!(
        format!("{ty} expected = {cmp};")
        format!(
            "while (!atomic_compare_exchange_weak_explicit({ptr}, &expected, {value}, memory_order_relaxed, memory_order_relaxed) && expected == {cmp}) {{}}"
        )
        "return expected;"
    )
});

metal_op_with_out!(AtomicIAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_add_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicFAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_add_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicISubOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_sub_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicFSubOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_sub_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicSMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_min_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicUMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_min_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicFMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_min_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicSMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_max_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicUMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_max_explicit({ptr}, {value}, memory_order_relaxed)")
});
metal_op_with_out!(AtomicFMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_max_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicAndOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_and_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicOrOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_or_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicXorOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_xor_explicit({ptr}, {value}, memory_order_relaxed)")
});
