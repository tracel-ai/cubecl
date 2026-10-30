//! CUDA's C++ atomic APIs are a mess of inconsistency. Old-style APIs use weird types, newer
//! `std::atomic` APIs don't support many types. So for ops where type support is complex, we lower
//! the atomics directly to PTX. This allows a consistent API across all the different type and
//! vectorization options, and significantly extends the interface that can be accessed from C++.

use cubecl_core::{
    frontend::reinterpret_value,
    ir::{Scope, dialect::atomic::*, interfaces::TypedExt, prelude::*, types::VectorType},
};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    value::Value,
};

use crate::{
    cuda::{
        packed_ops::packable,
        ptx::InlinePtxOp,
        ty::{BFloat16x2Type, Float16x2Type},
    },
    shared::{lowering::LowerOpAfterUnroll, ty::TypedExtCPP},
    target::Cuda,
};

fn atom_vec(ctx: &Context, val: impl Typed) -> &'static str {
    match val.vector_size(ctx) {
        1 => "",
        2 => ".v2",
        4 => ".v4",
        8 => ".v8",
        _ => unreachable!(),
    }
}

fn atom_ftz(ctx: &Context, val: impl Typed) -> &'static str {
    if val.is_half(ctx) || val.is_half2(ctx) {
        ".noftz"
    } else {
        ""
    }
}

// Signed only matters for cmp, addition is signless. And it's not supported for `s64`, only `u64`
fn atom_ty(ctx: &Context, val: impl Typed) -> &'static str {
    let scalar_ty = val.scalar_ty(ctx);
    if scalar_ty.is_float64(ctx) {
        "f64"
    } else if scalar_ty.is_float32(ctx) {
        "f32"
    } else if scalar_ty.is_float16(ctx) {
        "f16"
    } else if scalar_ty.deref(ctx).is::<Float16x2Type>() {
        "f16x2"
    } else if scalar_ty.is_bfloat16(ctx) {
        "bf16"
    } else if scalar_ty.deref(ctx).is::<BFloat16x2Type>() {
        "bf16x2"
    } else if scalar_ty.is_int_of_width(ctx, 64) {
        "u64"
    } else if scalar_ty.is_int_of_width(ctx, 32) {
        "u32"
    } else {
        panic!("Unsupported type")
    }
}

fn atom_ty_cmp(ctx: &Context, val: impl Typed) -> &'static str {
    let scalar_ty = val.scalar_ty(ctx);
    if scalar_ty.is_int_of_width(ctx, 64) && scalar_ty.is_signed_int(ctx) {
        "s64"
    } else if scalar_ty.is_int_of_width(ctx, 32) && scalar_ty.is_signed_int(ctx) {
        "s32"
    } else {
        atom_ty(ctx, val)
    }
}

// Reinterpet f16 etc
fn as_registers(scope: &Scope, val: Value) -> Value {
    let vec = val.vector_size(scope.ctx());
    let u16 = IntegerType::get(scope.ctx(), 16, Signedness::Unsigned).to_handle();
    let u32 = IntegerType::get(scope.ctx(), 32, Signedness::Unsigned).to_handle();
    if vec > 1 && val.is_half(scope.ctx()) {
        let vec_ty = VectorType::get(scope.ctx(), u16, vec);
        reinterpret_value(scope, val, vec_ty.to_handle())
    } else if vec > 1 && val.is_half2(scope.ctx()) {
        let vec_ty = VectorType::get(scope.ctx(), u32, vec);
        reinterpret_value(scope, val, vec_ty.to_handle())
    } else if val.is_half(scope.ctx()) {
        reinterpret_value(scope, val, u16)
    } else if val.is_half2(scope.ctx()) {
        reinterpret_value(scope, val, u32)
    } else {
        val
    }
}

macro_rules! atomic_binop {
    ($ty: ty, $op: literal, $atom_ty: ident) => {
        #[op_interface_impl]
        impl LowerOpAfterUnroll<Cuda> for $ty {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let ctx = scope.ctx_mut();
                let ptr = self.ptr(ctx);
                let value = self.value(ctx);
                let out_ty = self.get_result(ctx).get_type(ctx);

                let vec = atom_vec(ctx, value);
                let ftz = atom_ftz(ctx, value);
                let ty = $atom_ty(ctx, value);
                let value = as_registers(scope, value);

                let ptx = format!("atom.relaxed.{}{ftz}{vec}.{ty} $0, [$1], $2;", $op);
                let op = InlinePtxOp::new_volatile(
                    ctx,
                    Some(value.get_type(ctx)),
                    ptx,
                    vec![ptr, value],
                );
                scope.register(&op);
                vec![reinterpret_value(scope, op.result(ctx).unwrap(), out_ty)]
            }
        }
    };
}

/// An add whose old value nothing reads, of a scalar: `red`, which hands nothing back, so the add
/// issues and the thread moves on, where `atom` waits for the old value to return. A vector add
/// keeps `atom`, whose form this lowering already states.
macro_rules! atomic_add {
    ($ty: ty) => {
        #[op_interface_impl]
        impl LowerOpAfterUnroll<Cuda> for $ty {
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let ctx = scope.ctx_mut();
                let ptr = self.ptr(ctx);
                let value = self.value(ctx);
                let result = self.get_result(ctx);
                let out_ty = result.get_type(ctx);
                let vec = atom_vec(ctx, value);
                let ftz = atom_ftz(ctx, value);
                let ty = atom_ty(ctx, value);
                let unread = result.num_uses(ctx) == 0 && vec.is_empty();
                let registers = as_registers(scope, value);
                if unread {
                    let ptx = format!("red.relaxed.add{ftz}.{ty} [$0], $1;");
                    let op = InlinePtxOp::new_volatile(ctx, None, ptx, vec![ptr, registers]);
                    op.set_clobbers(ctx, vec!["memory".to_string()]);
                    scope.register(&op);
                    // Nothing reads the result; the value added stands in for it.
                    return vec![value];
                }
                let ptx = format!("atom.relaxed.add{ftz}{vec}.{ty} $0, [$1], $2;");
                let op = InlinePtxOp::new_volatile(
                    ctx,
                    Some(registers.get_type(ctx)),
                    ptx,
                    vec![ptr, registers],
                );
                scope.register(&op);
                vec![reinterpret_value(scope, op.result(ctx).unwrap(), out_ty)]
            }
        }
    };
}

packable!(AtomicFAddOp);
packable!(AtomicFMinOp);
packable!(AtomicFMaxOp);

atomic_add!(AtomicIAddOp);
atomic_add!(AtomicFAddOp);
atomic_binop!(AtomicSMinOp, "min", atom_ty_cmp);
atomic_binop!(AtomicUMinOp, "min", atom_ty_cmp);
atomic_binop!(AtomicFMinOp, "min", atom_ty_cmp);
atomic_binop!(AtomicSMaxOp, "max", atom_ty_cmp);
atomic_binop!(AtomicUMaxOp, "max", atom_ty_cmp);
atomic_binop!(AtomicFMaxOp, "max", atom_ty_cmp);
