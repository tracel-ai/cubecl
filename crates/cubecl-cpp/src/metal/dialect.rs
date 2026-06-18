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
    format!("os_log_default.log({format_string:?}{})", args)
});

metal_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane => "simdgroup_barrier(mem_flags::mem_none);\n",
        SyncScope::Cube => "threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        SyncScope::Device => "threadgroup_barrier(mem_flags::mem_device);\n",
    }
    .into()
});

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
// pub struct MslDialect {}

// // Base dialect

// impl Dialect for MslDialect {
//     type Architecture = MetalArchitecture;
// }

// // Includes

// impl DialectIncludes<Self> for MslDialect {
//     type Extension = Extension;

//     fn compile_includes(f: &mut std::fmt::Formatter<'_>, _flags: &Flags) -> std::fmt::Result {
//         write!(
//             f,
//             "
// #include <metal_stdlib>
// using namespace metal;
// "
//         )?;
//         Ok(())
//     }

//     fn compile_extensions(
//         f: &mut std::fmt::Formatter<'_>,
//         extensions: &[Self::Extension],
//     ) -> std::fmt::Result {
//         for extension in extensions {
//             match extension {
//                 Extension::SafeTanh(item) => format_safe_tanh(f, ctx, *item)?,
//                 Extension::NoExtension => {}
//             }
//         }
//         Ok(())
//     }
// }

// // Types

// impl DialectTypes<Self> for MslDialect {
//     fn compile_type_definitions(
//         f: &mut std::fmt::Formatter<'_>,
//         items: &std::collections::HashSet<crate::shared::Item<Self>>,
//         scalars: &[(Elem<Self>, usize)],
//         info: &cubecl_core::Info,
//         flags: &Flags,
//     ) -> std::fmt::Result {
//         for item in items.iter() {
//             if let Item::Vector(inner, vectorization) = item {
//                 let alignment = item.size();
//                 if *vectorization > 1 {
//                     write!(
//                         f,
//                         "
// struct alignas({alignment}) {item} {{"
//                     )?;

//                     for i in 0..*vectorization {
//                         write!(
//                             f,
//                             "
//     {inner} i_{i};"
//                         )?;
//                     }

//                     f.write_str("\n};\n")?;
//                 }
//             }
//         }

//         shared::type_info_definition_sized(f, info, scalars, flags.address_type)?;
//         Ok(())
//     }

//     fn compile_shared_memory_declaration(
//         f: &mut std::fmt::Formatter<'_>,
//         shared: &SharedMemory<Self>,
//     ) -> std::fmt::Result {
//         let SharedMemory { ptr, offset, .. } = shared;
//         let ptr_ty = ptr.item();
//         let size_bytes = shared.size();
//         writeln!(f, "// Shared value size: {size_bytes} bytes")?;
//         writeln!(
//             f,
//             "{ptr_ty} {ptr} = reinterpret_cast<{ptr_ty}>(&dynamic_shared_mem[{offset}]);"
//         )
//     }
// }

// // Kernel argument bindings

// impl DialectBindings<Self> for MslDialect {
//     fn compile_bindings_body(
//         f: &mut std::fmt::Formatter<'_>,
//         body: &shared::Body<Self>,
//     ) -> std::fmt::Result {
//         if !body.shared_memories.is_empty() {
//             let size = body
//                 .shared_memories
//                 .iter()
//                 .map(|it| it.offset + it.size())
//                 .max()
//                 .unwrap();

//             writeln!(f, "threadgroup uchar dynamic_shared_mem[{size}];",)?;
//         }
//         if body.info_by_ptr && body.has_dynamic_meta {
//             let address_space = AddressSpace::ConstDevice;
//             writeln!(f, "const {address_space} info_st& info = *info_ptr;")?;
//             // Could use `info_ptr + 1` but that seems dirty, so use manual `sizeof` instead
//             writeln!(
//                 f,
//                 "const {address_space} {addr}* dynamic_meta = reinterpret_cast<const {address_space} {addr}*>(
//                     reinterpret_cast<const {address_space} char*>(info_ptr) + sizeof(info_st)
//                 );\n",
//                 addr = body.address_type,
//             )?;
//         }
//         Ok(())
//     }
// }

// // Cube builtins dialect

// impl DialectCubeBuiltins<Self> for MslDialect {
//     /// Depending on the dialect available built-ins the
//     /// inclusion rules might change.
//     /// For instance in metal we have a built-in for the Unit plane position
//     /// so we don't rely on other builtins.
//     fn builtin_rules(flags: &CubeIndexFlags) -> CubeIndexFlags {
//         let absolute_pos = flags.absolute_pos;
//         let cube_count = flags.cube_count;
//         let cube_dim = flags.cube_dim;
//         let cube_pos = flags.cube_pos;
//         let plane_index = flags.plane_pos;
//         let unit_pos = flags.unit_pos;
//         let absolute_pos_tuple = flags.absolute_pos_tuple || absolute_pos;
//         let cube_count_tuple = flags.cube_count_tuple || cube_count || cube_pos || absolute_pos;
//         let cube_dim_tuple = flags.cube_dim_tuple || cube_dim || absolute_pos;
//         let cube_pos_tuple = flags.cube_pos_tuple || cube_pos;
//         let cluster_pos = flags.cluster_pos;
//         let plane_dim = flags.plane_dim || plane_index;
//         let unit_pos_plane = flags.unit_pos_plane || plane_index;
//         let unit_pos_tuple = flags.unit_pos_tuple || unit_pos;
//         CubeIndexFlags {
//             absolute_pos_tuple,
//             absolute_pos,
//             cube_count_tuple,
//             cube_count,
//             cube_dim_tuple,
//             cube_dim,
//             cube_pos_tuple,
//             cube_pos,
//             plane_dim,
//             plane_pos: plane_index,
//             unit_pos_tuple,
//             unit_pos,
//             unit_pos_plane,
//             cluster_pos,
//         }
//     }

//     fn compile_absolute_pos_tuple_computation(
//         _f: &mut std::fmt::Formatter<'_>,
//     ) -> std::fmt::Result {
//         // no need to compute it on metal as there is y a built-in for it
//         Ok(())
//     }

//     fn compile_unit_pos_computation(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         // no need to compute it on metal as there is y a built-in for it
//         Ok(())
//     }
// }

// Coop Matrices dialect

pub fn supported_cmma_combinations_metal(_arch: &MetalArchitecture) -> SupportedMmaCombinations {
    let types = vec![
        (
            ElemType::Float(FloatKind::F16).into(),
            ElemType::Float(FloatKind::F16).into(),
            ElemType::Float(FloatKind::F16).into(),
        ),
        (
            ElemType::Float(FloatKind::F16).into(),
            ElemType::Float(FloatKind::F16).into(),
            ElemType::Float(FloatKind::F32).into(),
        ),
        (
            ElemType::Float(FloatKind::BF16).into(),
            ElemType::Float(FloatKind::BF16).into(),
            ElemType::Float(FloatKind::BF16).into(),
        ),
        (
            ElemType::Float(FloatKind::F32).into(),
            ElemType::Float(FloatKind::F32).into(),
            ElemType::Float(FloatKind::F32).into(),
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
