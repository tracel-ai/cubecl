macro_rules! hip_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Hip> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use hip_op;

macro_rules! hip_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Hip> for $ty {
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
pub(super) use hip_op_with_out;

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
// pub struct HipDialect {}

// // Base dialect

// impl Dialect for HipDialect {
//     type Architecture = AMDArchitecture;
// }

// // Includes

// impl DialectIncludes<Self> for HipDialect {
//     type Extension = Extension;

//     fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
//         f.write_str("#include <hip/hip_runtime.h>\n")?;
//         if flags.elem_bf16 {
//             f.write_str("#include <hip/hip_bf16.h>\n")?;
//         }
//         if flags.elem_f16 {
//             f.write_str("#include <hip/hip_fp16.h>\n")?;
//         }
//         if flags.inst_wmma {
//             Self::compile_wmma_includes(f, flags)?;
//         }
//         Ok(())
//     }

//     fn compile_extensions(
//         f: &mut std::fmt::Formatter<'_>,
//         extensions: &[Self::Extension],
//     ) -> std::fmt::Result {
//         for extension in extensions {
//             match extension {
//                 Extension::NoExtension => {}
//                 Extension::Wmma(inst) => inst.format_wmma(f)?,
//             }
//         }
//         Ok(())
//     }
// }

// // Types

// impl DialectTypes<Self> for HipDialect {
//     fn compile_type_definitions(
//         f: &mut std::fmt::Formatter<'_>,
//         items: &HashSet<Item<Self>>,
//         scalars: &[(Elem<Self>, usize)],
//         info: &cubecl_core::Info,
//         flags: &Flags,
//     ) -> std::fmt::Result {
//         let mut items_deduplicated = HashSet::new();

//         for item in items {
//             let mut item = *item.value_ty();
//             match item {
//                 Item::NativeVector(..) => {
//                     continue;
//                 }
//                 Item::Atomic(inner) => {
//                     item = *inner;
//                 }
//                 _ => {}
//             }
//             items_deduplicated.insert(item);
//         }

//         shared::type_definitions(f)?;
//         shared::type_vectorized_definitions(f, &items_deduplicated)?;

//         shared::type_info_definition_sized(f, info, scalars, flags.address_type)?;

//         if flags.inst_wmma {
//             Self::compile_wmma_type_definitions(f, flags)?;
//         }

//         Ok(())
//     }
// }

// // Kernel argument bindings

// impl DialectBindings<Self> for HipDialect {
//     fn compile_bindings_body(
//         f: &mut std::fmt::Formatter<'_>,
//         body: &shared::Body<Self>,
//     ) -> std::fmt::Result {
//         if !body.shared_memories.is_empty() {
//             let max_align = body
//                 .shared_memories
//                 .iter()
//                 .map(|smem| smem.smem.alignment)
//                 .max()
//                 .unwrap();
//             // The `__align__` instead of `alignas` is on purpose - the compiler is currently bugged
//             // with `extern __shared__ alignas` and doesn't properly parse it.
//             writeln!(
//                 f,
//                 "extern __shared__ __align__({max_align}) uchar dynamic_shared_mem[];"
//             )?;
//         }
//         Ok(())
//     }
// }

// // Cube builtins dialect

// impl DialectCubeBuiltins<Self> for HipDialect {}
