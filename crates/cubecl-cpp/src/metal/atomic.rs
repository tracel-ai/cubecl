use cubecl_core::ir::{dialect::atomic::*, prelude::*};

use crate::{
    metal::{metal_op, metal_op_with_out},
    shared::CppValue,
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

metal_op_with_out!(AtomicCompareExchangeWeakOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let cmp = op.cmp(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!(
        "atomic_compare_exchange_weak_explicit({ptr}, &{cmp}, {value}, memory_order_relaxed, memory_order_relaxed)"
    )
});

metal_op_with_out!(AtomicAddOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_add_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicSubOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_sub_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_fetch_min_explicit({ptr}, {value}, memory_order_relaxed)")
});

metal_op_with_out!(AtomicMaxOp, |op, ctx| {
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
