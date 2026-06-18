use core::fmt::Display;

use cubecl_core::ir::{
    dialect::synchronization::SyncAsyncProxyOp, interfaces::TypedExt, prelude::*, types::VectorType,
};
use itertools::Itertools;
use pliron::{
    builtin::attributes::{StringAttr, UnitAttr},
    printable::Printable,
};

use crate::shared::{
    CppValue, scoped_block,
    ty::{AddressSpace, PointerType, TypeExtCPP},
};

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
                let op = $crate::cuda::dialect::InlinePtxOp::new(
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

cuda_op!(SyncAsyncProxyOp, |_, _| {
    "cuda::device::experimental::fence_proxy_async_shared_cta();".into()
});

/// Inline PTX. Restricted to zero or one results because C++ semantics are too hard otherwise.
/// Note that this does not *directly* map to PTX, because it actually destructures vectors to PTX
/// vector expressions automatically. This means more than one register can be returned if it's part
/// of a vector expression. To denote the difference, the syntax uses `$0`, `$1` etc for Pliron
/// values, as opposed to the usual `%0`, `%1` etc for the PTX registers.
#[pliron_op(name = "cuda.inline_ptx", format, attributes = (ptx: StringAttr, volatile: UnitAttr), verifier = "succ")]
pub struct InlinePtxOp;

impl InlinePtxOp {
    pub fn new(
        ctx: &mut Context,
        result_ty: Option<TypeHandle>,
        ptx: impl Display,
        inputs: Vec<Value>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_ty.into_iter().collect(),
            inputs,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_ptx(ctx, ptx.to_string().into());
        op
    }

    pub fn new_volatile(
        ctx: &mut Context,
        result_ty: Option<TypeHandle>,
        ptx: impl Display,
        inputs: Vec<Value>,
    ) -> Self {
        let op = Self::new(ctx, result_ty, ptx, inputs);
        op.set_attr_volatile(ctx, UnitAttr::new());
        op
    }

    pub fn raw_ptx(&self, ctx: &Context) -> String {
        self.get_attr_ptx(ctx).unwrap().clone().into()
    }

    pub fn is_volatile(&self, ctx: &Context) -> bool {
        self.get_attr_volatile(ctx).is_some()
    }

    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }

    pub fn result(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).results().next()
    }
}

macro_rules! ptx_block {
    ($($lines: expr)*) => {{
        let mut out = String::from("{\n\t");
        $(
            out.push_str(&$lines);
            out.push_str("\n\t");
        )*
        out.push_str("}");
        out
    }};
}
pub(crate) use ptx_block;

cuda_op!(InlinePtxOp, |op, ctx| {
    let mut ptx = op.raw_ptx(ctx);
    let result = op.result(ctx);
    let inputs = op.inputs(ctx);

    let mut ptx_idx = 0;
    let mut plir_idx = 0;

    if let Some(result) = result {
        ptx = insert_placeholders(ctx, &ptx, result.get_type(ctx), plir_idx, &mut ptx_idx);
        plir_idx += 1;
    }

    for input in inputs.iter() {
        ptx = insert_placeholders(ctx, &ptx, input.get_type(ctx), plir_idx, &mut ptx_idx);
        plir_idx += 1;
    }

    let out_regs = result
        .iter()
        .flat_map(|val| flatten_operand(ctx, *val, "="))
        .join(", ");
    let input_regs = inputs
        .iter()
        .flat_map(|val| flatten_operand(ctx, *val, ""))
        .join(", ");

    let volatile = if op.is_volatile(ctx) { "volatile" } else { "" };
    let asm = format!("asm {volatile}({ptx:?} : {out_regs} : {input_regs});",);

    if let Some(result) = result {
        let block = scoped_block!(
            format!("{} result;", result.get_type(ctx).to_cpp(ctx))
            asm
            format!("return result;")
        );
        format!("{} = {block};", result.fmt_left(ctx))
    } else {
        asm
    }
});

fn flatten_operand(ctx: &Context, val: Value, prefix: &str) -> Vec<String> {
    if val.get_type(ctx).deref(ctx).is::<VectorType>() {
        let vec = val.vector_size(ctx);
        let constraint = infer_constraint_letter(ctx, val.scalar_ty(ctx));
        (0..vec)
            .map(|i| format!(r#""{prefix}{constraint}"({}.i_{i})"#, val.name(ctx)))
            .collect()
    } else {
        let constraint = infer_constraint_letter(ctx, val.get_type(ctx));
        vec![format!(r#""{prefix}{constraint}"({})"#, val.name(ctx))]
    }
}

fn insert_placeholders(
    ctx: &Context,
    ptx: &str,
    ty: TypeHandle,
    plir_idx: usize,
    ptx_idx: &mut usize,
) -> String {
    let pat = format!("${plir_idx}");
    if !ptx.contains(&pat) {
        panic!("Tried substituting argument {pat} in PTX, but it wasn't found.")
    }
    let substitute = if ty.deref(ctx).is::<VectorType>() {
        let vec = ty.vector_size(ctx);
        let mut placeholders = (0..vec).map(|i| format!("%{}", *ptx_idx + i));
        let substitute = format!("{{{}}}", placeholders.join(", "));
        *ptx_idx += vec;
        substitute
    } else {
        let placeholder = format!("%{ptx_idx}");
        *ptx_idx += 1;
        placeholder
    };
    ptx.replace(&pat, &substitute)
}

fn infer_constraint_letter(ctx: &Context, ty: TypeHandle) -> char {
    if ty.is_bool(ctx) {
        'b'
    } else if ty.is_int_of_width(ctx, 16) || ty.is_uint_of_width(ctx, 16) {
        'h'
    } else if ty.is_int_of_width(ctx, 32) || ty.is_uint_of_width(ctx, 32) {
        'r'
    } else if ty.is_int_of_width(ctx, 64) || ty.is_uint_of_width(ctx, 64) {
        'l'
    } else if ty.is_float32(ctx) {
        'f'
    } else if ty.is_float64(ctx) {
        'd'
    } else if let Some(ptr) = ty.deref(ctx).downcast_ref::<PointerType>() {
        // Shared address spaces is addressed with 32-bit pointers.
        if ptr.address_space == AddressSpace::Shared {
            'r'
        } else {
            'l'
        }
    } else {
        panic!(
         "The register type could not be deduced from Pliron type. The type {} is not supported. 
Supported types are: bool, i16, i32, i64, f32, f64, pointers.
Please use cube.reinterpret_cast if you have different type.
See the constraints from here: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints",
        ty.disp(ctx));
    }
}

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
// pub struct CudaDialect {}

// impl Dialect for CudaDialect {
//     type Architecture = CudaArchitecture;
// }

// impl DialectIncludes<Self> for CudaDialect {
//     type Extension = Extension;

//     fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags<Self>) -> std::fmt::Result {
//         f.write_str("#include <cuda_runtime.h>\n")?;
//         if flags.elem_fp4 {
//             f.write_str("#include <cuda_fp4.h>\n")?;
//         }
//         if flags.elem_fp6 {
//             f.write_str("#include <cuda_fp6.h>\n")?;
//         }
//         if flags.elem_fp8 {
//             f.write_str("#include <cuda_fp8.h>\n")?;
//         }
//         if flags.elem_bf16 {
//             f.write_str("#include <cuda_bf16.h>\n")?;
//         }
//         if flags.elem_f16 {
//             f.write_str("#include <cuda_fp16.h>\n")?;
//         }

//         // tf32 conversion function is in mma header
//         if flags.inst_wmma || flags.elem_tf32 {
//             Self::compile_wmma_includes(f, flags)?;
//         }

//         if flags.op_barrier || flags.inst_tma || flags.indexes.cluster_pos {
//             f.write_str("#include <cooperative_groups.h>\n")?;
//             f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
//             f.write_str("#include <cuda/barrier>\n")?;
//         }
//         if flags.inst_ptx_wrappers {
//             f.write_str("#include <cuda/ptx>\n")?;
//         }
//         if flags.inst_tma {
//             f.write_str(
//                 "typedef struct CUtensorMap_st {
// alignas(64) unsigned long long int opaque[16];
// } CUtensorMap;\n",
//             )?;
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
//         if body.info_by_ptr {
//             f.write_str("const info_st& info = *info_ptr;\n")?;
//             // Could use `info_ptr + 1` but that seems dirty, so use manual `sizeof` instead
//             writeln!(
//                 f,
//                 "const {addr}* dynamic_meta = reinterpret_cast<const {addr}*>(
//                     reinterpret_cast<const char*>(info_ptr) + sizeof(info_st)
//                 );\n",
//                 addr = body.address_type,
//             )?;
//         }
//         Ok(())
//     }
// }
