use cubecl_core::ir::{dialect::atomic::*, prelude::*};

use crate::{
    opencl::{opencl_op, opencl_op_with_out},
    shared::{CppValue, ty::TypeExtCPP},
};

// OpenCL C 3.0's C11 atomics, relaxed, at device scope (the default of the `_explicit` forms);
// float add / min / max from `cl_ext_float_atomics`.

opencl_op_with_out!(AtomicLoadOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    format!("atomic_load_explicit({ptr}, memory_order_relaxed)")
});

opencl_op!(AtomicStoreOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomic_store_explicit({ptr}, {value}, memory_order_relaxed);\n")
});

// Answers with the value it saw, as CUDA's `atomicCAS`; a spurious failure (it saw `cmp`) retries.
opencl_op_with_out!(AtomicCompareExchangeWeakOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let cmp = op.cmp(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let ty = op.cmp(ctx).get_type(ctx).to_cpp(ctx);
    format!(
        "({{ {ty} expected = {cmp}; while (!atomic_compare_exchange_weak_explicit({ptr}, &expected, {value}, memory_order_relaxed, memory_order_relaxed) && expected == {cmp}) {{}} expected; }})"
    )
});

/// `atomic_<function>_explicit(ptr, value, memory_order_relaxed)`.
macro_rules! relaxed {
    ($op: ty, $function: literal) => {
        opencl_op_with_out!($op, |op, ctx| format!(
            concat!(
                "atomic_",
                $function,
                "_explicit({}, {}, memory_order_relaxed)"
            ),
            op.ptr(ctx).name(ctx),
            op.value(ctx).name(ctx)
        ));
    };
}

relaxed!(AtomicExchangeOp, "exchange");
relaxed!(AtomicIAddOp, "fetch_add");
relaxed!(AtomicFAddOp, "fetch_add");
relaxed!(AtomicSMinOp, "fetch_min");
relaxed!(AtomicUMinOp, "fetch_min");
relaxed!(AtomicFMinOp, "fetch_min");
relaxed!(AtomicSMaxOp, "fetch_max");
relaxed!(AtomicUMaxOp, "fetch_max");
relaxed!(AtomicFMaxOp, "fetch_max");
relaxed!(AtomicAndOp, "fetch_and");
relaxed!(AtomicOrOp, "fetch_or");
relaxed!(AtomicXorOp, "fetch_xor");
