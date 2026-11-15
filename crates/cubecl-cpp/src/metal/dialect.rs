use super::arch::MetalArchitecture;
use crate::shared::{CppValue, SupportedMmaCombinations};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    dialect::{
        general::PrintfOp,
        synchronization::{SyncOp, SyncPlaneStorageOp, SyncScope},
    },
    features::MmaConfig,
};
use itertools::Itertools;

macro_rules! metal_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Metal> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use metal_op;

macro_rules! metal_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Metal> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                use cubecl_core::ir::prelude::*;
                use $crate::shared::CppValue;
                let op = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl);
                let out = self.get_result(ctx).fmt_left(ctx);
                format!("{out} = {op};\n")
            }
        }
    };
}
pub(super) use metal_op_with_out;

metal_op!(PrintfOp, |op, ctx| {
    let format_string = String::from(op.format_string(ctx).clone());
    let args = op.args(ctx);
    let args = args.iter().map(|it| format!(", {}", it.name(ctx))).join("");
    format!("os_log_default.log({format_string:?}{});\n", args)
});

metal_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        // The device flag is what makes the barrier order threadgroup memory too: measured on an
        // M2, with `mem_threadgroup` alone a unit's second store before the barrier sometimes
        // still had not reached the plane's other units after it, and seq_cst fences at simdgroup
        // or threadgroup scope around it did not change that.
        SyncScope::Plane => "simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n",
        SyncScope::Cube => "threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        // Three instructions, each measured on an M2 to be doing something. The barrier is the
        // cube's half, and on its own it leaves one threadgroup's writes unseen by the next. The
        // fence at `thread_scope_device` is the device half, and *which side of the barrier it
        // sits on* decides what it covers: with the fence only after, a store made by one unit
        // before the barrier still reached no other threadgroup, and the same kernel with one
        // more `threadgroup_barrier` in front of it did. So the release is stated before the
        // cube meets and the acquire after, and the barrier between them is what makes both the
        // whole cube's. `atomic_thread_fence` is MSL 3.2, the version this backend compiles at.
        // No memory is shared with another device, so the system is the device.
        SyncScope::Device | SyncScope::System => {
            "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);\n\
             threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n\
             atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);\n"
        }
        SyncScope::Unit => "",
    }
    .into()
});

// The simdgroup barrier's device flag is what orders the plane's storage writes for its own units.
metal_op!(SyncPlaneStorageOp, |_, _| {
    "simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n".into()
});

// Coop Matrices dialect

pub fn supported_cmma_combinations_metal(_arch: &MetalArchitecture) -> SupportedMmaCombinations {
    let types = vec![
        (
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F16),
        ),
        (
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
        ),
        (
            ElemType::Float(FloatKind::BF16),
            ElemType::Float(FloatKind::BF16),
            ElemType::Float(FloatKind::BF16),
        ),
        (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F32),
        ),
    ];
    types
        .into_iter()
        .map(|(a_type, b_type, cd_type)| MmaConfig {
            a_type,
            b_type,
            cd_type,
            m: 8,
            n: 8,
            k: 8,
        })
        .collect()
}

/// The cooperative matrices that run as Metal 4 tensor operations (`matmul2d` on one SIMD group):
/// every shape whose sides are 16 or 32, one of them 32, which is what the operation takes with
/// both inputs held in registers. The lowering refuses every other product.
pub fn supported_tensor_op_combinations_metal() -> SupportedMmaCombinations {
    let (f16, bf16, f32) = (FloatKind::F16, FloatKind::BF16, FloatKind::F32);
    let floats = [(f16, f16), (f16, f32), (bf16, bf16), (bf16, f32)]
        .map(|(ab, cd)| (ElemType::Float(ab), ElemType::Float(cd)));
    let integers = [ElemType::Int(IntKind::I8), ElemType::UInt(UIntKind::U8)]
        .map(|ab| (ab, ElemType::Int(IntKind::I32)));
    let sides = [16, 32];
    let shapes = sides
        .into_iter()
        .cartesian_product(sides)
        .cartesian_product(sides)
        .map(|((m, n), k)| (m, n, k))
        .filter(|&(m, n, k)| m == 32 || n == 32 || k == 32);
    floats
        .into_iter()
        .chain(integers)
        .cartesian_product(shapes)
        .map(|((ab, cd_type), (m, n, k))| MmaConfig {
            a_type: ab,
            b_type: ab,
            cd_type,
            m,
            n,
            k,
        })
        .collect()
}
