use cubecl_core::{
    self as cubecl,
    ir::{
        dialect::{atomic::*, synchronization::SyncScope},
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};
use num_traits::One;
use pliron::{context::Ptr, operation::Operation};

use crate::{
    shared::{lowering::LowerOp, scoped_block, shared_op_with_out, ty::TypeExtCPP},
    target::{CtxTarget, Target},
};

/// The CUDA and HIP function `name` for the atomic `op`: its `_system` variant when the atomic
/// synchronizes at system scope, which reaches the host and peer devices.
pub(crate) fn scoped_name(ctx: &Context, op: Ptr<Operation>, name: &str) -> String {
    match atomic_scope(ctx, op) {
        SyncScope::System => format!("{name}_system"),
        _ => name.into(),
    }
}

/// Refuse a system-scope atomic on a 16-byte value: no `_system` variant takes one.
fn wide_device_scope_only(ctx: &Context, op: Ptr<Operation>) {
    assert!(
        atomic_scope(ctx, op) != SyncScope::System,
        "no system-scope atomic exists for a 16-byte value"
    );
}

#[cube]
fn atomic_i_sub<T: Numeric + CubeNot, N: Size>(
    ptr: Atomic<Vector<T, N>>,
    value: Vector<T, N>,
    #[comptime] sync: SyncScope,
) -> Vector<T, N> {
    ptr.fetch_add_at(!value + Vector::one(), sync)
}

#[cube]
fn atomic_f_sub<T: Numeric + CubeNeg, N: Size>(
    ptr: Atomic<Vector<T, N>>,
    value: Vector<T, N>,
    #[comptime] sync: SyncScope,
) -> Vector<T, N> {
    ptr.fetch_add_at(-value, sync)
}

#[cube]
fn atomic_store<T: Numeric + CubeNeg, N: Size>(
    ptr: Atomic<Vector<T, N>>,
    value: Vector<T, N>,
    #[comptime] sync: SyncScope,
) {
    ptr.exchange_at(value, sync);
}

// The rewrites below build other atomics, which synchronize at the scope the original asked for.

#[op_interface_impl]
impl LowerOp for AtomicStoreOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let ptr = self.ptr(scope.ctx());
        let value = self.value(scope.ctx());
        let sync = atomic_scope(scope.ctx(), self.get_operation());
        scope.register_value_type::<T, S>(value);
        atomic_store::expand::<T, S>(scope, ptr.into(), value.into(), sync);
        vec![]
    }
}

macro_rules! lower_atomic_sub {
    ($ty: ty, $name: ident) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                ctx.target() != Target::Metal
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let ptr = self.ptr(scope.ctx());
                let value = self.value(scope.ctx());
                let sync = atomic_scope(scope.ctx(), self.get_operation());
                scope.register_value_type::<T, S>(value);
                vec![$name::expand::<T, S>(scope, ptr.into(), value.into(), sync).read_value(scope)]
            }
        }
    };
}

lower_atomic_sub!(AtomicISubOp, atomic_i_sub);
lower_atomic_sub!(AtomicFSubOp, atomic_f_sub);

shared_op_with_out!(AtomicLoadOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx);
    let uint_ty = match out_ty.size(ctx) {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "uint64_t",
        16 => {
            wide_device_scope_only(ctx, op.get_operation());
            return scoped_block! {
                format!("{} tmp;", out_ty.to_cpp(ctx))
                format!("__nv_atomic_load({ptr}, &tmp, __NV_ATOMIC_RELAXED);")
                "return tmp;"
            };
        }
        _ => unreachable!(),
    };
    // A volatile load is relaxed at system scope on CUDA (PTX gives `ld.volatile` the semantics
    // of `ld.relaxed.sys`). HIP makes no such promise, so a system-scope load says so.
    if ctx.target() == Target::Hip && atomic_scope(ctx, op.get_operation()) == SyncScope::System {
        return scoped_block! {
            format!("const {uint_ty} tmp = __hip_atomic_load(reinterpret_cast<{uint_ty} const*>({ptr}), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);")
            format!("return reinterpret_cast<const {}&>(tmp);", out_ty.to_cpp(ctx))
        };
    }
    scoped_block! {
        format!("volatile {uint_ty} const* tmp = reinterpret_cast<volatile {uint_ty} const*>({ptr});")
        format!("const {uint_ty} tmp_2 = *tmp;")
        format!("return reinterpret_cast<const {}&>(tmp_2);", out_ty.to_cpp(ctx))
    }
});

shared_op_with_out!(AtomicExchangeOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx);
    let uint_ty = match out_ty.size(ctx) {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "uint64_t",
        16 => {
            wide_device_scope_only(ctx, op.get_operation());
            return format!("atomicExch({ptr}, {value})");
        }
        _ => unreachable!(),
    };
    let exch = scoped_name(ctx, op.get_operation(), "atomicExch");
    let ptr = format!("reinterpret_cast<{uint_ty}*>({ptr})");
    let value = format!("reinterpret_cast<const {uint_ty}&>({value})");
    scoped_block! {
        format!("const {uint_ty} tmp = {exch}({ptr}, {value});")
        format!("return reinterpret_cast<const {}&>(tmp);", out_ty.to_cpp(ctx))
    }
});

shared_op_with_out!(AtomicCompareExchangeWeakOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let cmp = op.cmp(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx);
    let uint_ty = match out_ty.size(ctx) {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "uint64_t",
        16 => {
            wide_device_scope_only(ctx, op.get_operation());
            return format!("atomicCAS({ptr}, {cmp}, {value})");
        }
        _ => unreachable!(),
    };
    let cas = scoped_name(ctx, op.get_operation(), "atomicCAS");
    let ptr = format!("reinterpret_cast<{uint_ty}*>({ptr})");
    let cmp = format!("reinterpret_cast<const {uint_ty}&>({cmp})");
    let value = format!("reinterpret_cast<const {uint_ty}&>({value})");
    scoped_block! {
        format!("const {uint_ty} tmp = {cas}({ptr}, {cmp}, {value});")
        format!("return reinterpret_cast<const {}&>(tmp);", out_ty.to_cpp(ctx))
    }
});

shared_op_with_out!(AtomicSMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMin");
    format!("{name}({ptr}, {value})")
});
shared_op_with_out!(AtomicUMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMin");
    format!("{name}({ptr}, {value})")
});
shared_op_with_out!(AtomicFMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMin");
    format!("{name}({ptr}, {value})")
});

shared_op_with_out!(AtomicSMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMax");
    format!("{name}({ptr}, {value})")
});
shared_op_with_out!(AtomicUMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMax");
    format!("{name}({ptr}, {value})")
});
shared_op_with_out!(AtomicFMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicMax");
    format!("{name}({ptr}, {value})")
});

shared_op_with_out!(AtomicAndOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicAnd");
    format!("{name}({ptr}, {value})")
});

shared_op_with_out!(AtomicOrOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicOr");
    format!("{name}({ptr}, {value})")
});

shared_op_with_out!(AtomicXorOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let name = scoped_name(ctx, op.get_operation(), "atomicXor");
    format!("{name}({ptr}, {value})")
});
