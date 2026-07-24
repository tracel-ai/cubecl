use cubecl_core::ir::{dialect::synchronization::SyncAsyncProxyOp, prelude::*};

use crate::{shared::signature::op_includes, target::Cuda};

macro_rules! cuda_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use cuda_op;

macro_rules! cuda_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
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
pub(super) use cuda_op_with_out;

macro_rules! ptx_with_out {
    ($ty: ty, $ptx: expr, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::shared::lowering::LowerOp<$crate::target::Cuda> for $ty {
            fn should_lower(&self, ctx: &pliron::context::Context) -> bool {
                $crate::shared::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }
            fn lower(&self, scope: &cubecl_core::ir::Scope) -> Vec<pliron::value::Value> {
                use cubecl_core::ir::dialect::base::OperationPtrExt;
                use pliron::{op::Op, r#type::Typed};
                let ctx = scope.ctx_mut();
                let ptx = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $ptx);
                let op = $crate::cuda::ptx::InlinePtxOp::new(
                    ctx,
                    Some(self.get_result(ctx).get_type(ctx)),
                    ptx,
                    self.get_operation().operands(ctx),
                );
                scope.register(&op);
                vec![op.result(ctx).unwrap()]
            }
        }
    };
    ($ty: ty, $ptx: expr) => {
        ptx_with_out!($ty, $ptx, |_, _| true);
    };
}
pub(super) use ptx_with_out;

op_includes!(Cuda, [SyncAsyncProxyOp] => "cuda/barrier");

cuda_op!(SyncAsyncProxyOp, |_, _| {
    "cuda::device::experimental::fence_proxy_async_shared_cta();".into()
});

//     fn compile_extensions(
//         f: &mut std::fmt::Formatter<'_>,
//         extensions: &[Self::Extension],
//     ) -> std::fmt::Result {
//         for extension in extensions {
//             match extension {
//                 Extension::NoExtension => {}
//                 Extension::Mma(mma) => write!(f, "{}", mma.format_extension(ctx))?,
//             }
//         }
//         Ok(())
//     }
// }

// // Types

// impl DialectTypes<Self> for CudaDialect {
//     fn compile_type_definitions(
//         f: &mut std::fmt::Formatter<'_>,
//         items: &HashSet<Item<Self>>,
//         scalars: &[(Elem<Self>, usize)],
//         info: &cubecl_core::Info,
//         flags: &Flags,
//     ) -> std::fmt::Result {
//         // All FP4/FP6/FP8 elems map to the same type, so we need to deduplicate them
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
//             match item.elem() {
//                 Elem::FP4(_) => {
//                     item = item.with_elem(Elem::FP4(FP4Kind::E2M1));
//                 }
//                 Elem::FP4x2(_) => {
//                     item = item.with_elem(Elem::FP4x2(FP4Kind::E2M1));
//                 }
//                 Elem::FP6(_) => {
//                     item = item.with_elem(Elem::FP6(FP6Kind::E2M3));
//                 }
//                 Elem::FP6x2(_) => {
//                     item = item.with_elem(Elem::FP6x2(FP6Kind::E2M3));
//                 }
//                 Elem::FP8(_) => {
//                     item = item.with_elem(Elem::FP8(FP8Kind::E4M3));
//                 }
//                 Elem::FP8x2(_) => {
//                     item = item.with_elem(Elem::FP8x2(FP8Kind::E4M3));
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

//     fn compile_polyfills(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
//         if flags.inst_tma_im2col {
//             writeln!(f, "{TMA_LOAD_IM2COL}")?;
//         }
//         if flags.inst_async_copy {
//             writeln!(f, "{COPY_ASYNC}")?;
//         }
//         Ok(())
//     }
// }

// // Kernel argument bindings

// impl DialectBindings<Self> for CudaDialect {
//     fn compile_bindings_body(
//         f: &mut std::fmt::Formatter<'_>,
//         body: &shared::Body<Self>,
//     ) -> std::fmt::Result {
//         if !body.shared_memories.is_empty() {
//             let max_align = body
//                 .shared_memories
//                 .iter()
//                 .map(|smem| smem.align)
//                 .max()
//                 .unwrap();
//             // The `__align__` instead of `alignas` is on purpose - the compiler is currently bugged
//             // with `extern __shared__ alignas` and doesn't properly parse it.
//             writeln!(
//                 f,
//                 "extern __shared__ __align__({max_align}) uint8 dynamic_shared_mem[];"
//             )?;
//         }
//         Ok(())
//     }
// }
