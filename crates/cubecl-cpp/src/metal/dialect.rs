use super::arch::MetalArchitecture;
use crate::shared::{CppValue, SupportedMmaCombinations};
use cubecl_core::ir::{
    ElemType, FloatKind,
    dialect::{
        general::PrintfOp,
        synchronization::{SyncOp, SyncScope},
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
        SyncScope::Plane => "simdgroup_barrier(mem_flags::mem_none);\n",
        SyncScope::Cube => "threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        SyncScope::Device => "threadgroup_barrier(mem_flags::mem_device);\n",
        SyncScope::Unit => "",
    }
    .into()
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
