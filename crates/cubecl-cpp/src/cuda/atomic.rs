//! CUDA's C++ atomic APIs are a mess of inconsistency. Old-style APIs use weird types, newer
//! `std::atomic` APIs don't support many types. So for ops where type support is complex, we lower
//! the atomics directly to PTX. This allows a consistent API across all the different type and
//! vectorization options, and significantly extends the interface that can be accessed from C++.

use cubecl_core::{
    frontend::reinterpret_value,
    ir::{
        Scope,
        dialect::atomic::*,
        interfaces::TypedExt,
        prelude::*,
        types::{VectorType, scalar::UIntType},
    },
};
use pliron::value::Value;

use crate::{
    cuda::{
        ptx::InlinePtxOp,
        ty::{BFloat16x2Type, Float16x2Type},
    },
    shared::{lowering::LowerOp, ty::TypedExtCPP},
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
    } else if scalar_ty.is_int_of_width(ctx, 64) || scalar_ty.is_uint_of_width(ctx, 64) {
        "u64"
    } else if scalar_ty.is_int_of_width(ctx, 32) || scalar_ty.is_uint_of_width(ctx, 32) {
        "u32"
    } else {
        panic!("Unsupported type")
    }
}

fn atom_ty_cmp(ctx: &Context, val: impl Typed) -> &'static str {
    let scalar_ty = val.scalar_ty(ctx);
    if scalar_ty.is_int_of_width(ctx, 64) {
        "s64"
    } else if scalar_ty.is_int_of_width(ctx, 32) {
        "s32"
    } else {
        atom_ty(ctx, val)
    }
}

// Reinterpet f16 etc
fn as_registers(scope: &Scope, val: Value) -> Value {
    let vec = val.vector_size(scope.ctx());
    let u16 = UIntType::get(scope.ctx(), 16).to_handle();
    let u32 = UIntType::get(scope.ctx(), 32).to_handle();
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
        impl LowerOp<Cuda> for $ty {
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

atomic_binop!(AtomicAddOp, "add", atom_ty);
atomic_binop!(AtomicMinOp, "min", atom_ty_cmp);
atomic_binop!(AtomicMaxOp, "max", atom_ty_cmp);
