use cubecl_core::{
    self as cubecl,
    ir::{dialect::atomic::*, interfaces::TypedExt, prelude::*},
    prelude::*,
};

use crate::{
    shared::{
        binary::lower_binop, lowering::LowerOp, scoped_block, shared_op_with_out, ty::TypeExtCPP,
    },
    target::{CtxTarget, Target},
};

#[cube]
fn atomic_sub<T: Numeric + CubeNeg, N: Size>(
    ptr: Atomic<Vector<T, N>>,
    value: Vector<T, N>,
) -> Vector<T, N> {
    ptr.fetch_add(-value)
}

#[cube]
fn atomic_store<T: Numeric + CubeNeg, N: Size>(ptr: Atomic<Vector<T, N>>, value: Vector<T, N>) {
    ptr.exchange(value);
}

#[op_interface_impl]
impl LowerOp for AtomicStoreOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let ptr = self.ptr(scope.ctx());
        let value = self.value(scope.ctx());
        scope.register_value_type::<T, S>(value);
        atomic_store::expand::<T, S>(scope, ptr.into(), value.into());
        vec![]
    }
}

lower_binop!(AtomicSubOp, atomic_sub, |_, ctx| {
    ctx.target() != Target::Metal
});

shared_op_with_out!(AtomicLoadOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx);
    let uint_ty = match out_ty.size(ctx) {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "uint64_t",
        16 => {
            return scoped_block! {
                format!("{} tmp;", out_ty.to_cpp(ctx))
                format!("__nv_atomic_load({ptr}, &tmp, __NV_ATOMIC_RELAXED);")
                format!("return tmp;")
            };
        }
        _ => unreachable!(),
    };
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
            return format!("atomicExch({ptr}, {value})");
        }
        _ => unreachable!(),
    };
    let ptr = format!("reinterpret_cast<{uint_ty}*>({ptr})");
    let value = format!("reinterpret_cast<const {uint_ty}&>({value})");
    scoped_block! {
        format!("const {uint_ty} tmp = atomicExch({ptr}, {value});")
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
            return format!("atomicCAS({ptr}, {cmp}, {value})");
        }
        _ => unreachable!(),
    };
    let ptr = format!("reinterpret_cast<{uint_ty}*>({ptr})");
    let cmp = format!("reinterpret_cast<const {uint_ty}&>({cmp})");
    let value = format!("reinterpret_cast<const {uint_ty}&>({value})");
    scoped_block! {
        format!("const {uint_ty} tmp = atomicCAS({ptr}, {cmp}, {value});")
        format!("return reinterpret_cast<const {}&>(tmp);", out_ty.to_cpp(ctx))
    }
});

shared_op_with_out!(AtomicMinOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicMin({ptr}, {value})")
});

shared_op_with_out!(AtomicMaxOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicMax({ptr}, {value})")
});

shared_op_with_out!(AtomicAndOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicAnd({ptr}, {value})")
});

shared_op_with_out!(AtomicOrOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicOr({ptr}, {value})")
});

shared_op_with_out!(AtomicXorOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!("atomicXor({ptr}, {value})")
});
