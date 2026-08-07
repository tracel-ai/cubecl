use cubecl_core::ir::{
    attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, EntrypointInterface, FuncInterface},
    prelude::*,
};
use cubecl_opt::passes::alloc_shared_memory::AllocSharedOp;
use itertools::Itertools;
use pliron::{
    builtin::{
        ops::{FuncOp, ModuleOp},
        types::FunctionType,
    },
    dict_key,
};

use crate::{
    metal::{BuiltInAttr, metal_op},
    shared::{
        CppValue, branch::block_to_cpp, signature::LoadInfoOp, ty::TypeExtCPP, type_definitions,
    },
};

dict_key!(ATTR_BUILTIN_ATTRIBUTE, "metal_builtin");

const IMPORT: &str = "
#include <metal_stdlib>
using namespace metal;
";

metal_op!(ModuleOp, |op, ctx| {
    let mut out = IMPORT.to_string();
    type_definitions(&mut out, "long").unwrap();
    out.push_str(&block_to_cpp(ctx, op.get_body(ctx, 0)));
    out
});

metal_op!(FuncOp, |op, ctx| {
    let func_name = op.get_symbol_name(ctx);
    let ty = op.get_type(ctx).deref(ctx);
    let func_ty = ty.downcast_ref::<FunctionType>().unwrap();
    let return_ty = func_ty.res_types()[0].to_cpp(ctx);
    let attributes = if let Some(abi) = op.get_entrypoint_abi(ctx) {
        format!(
            r#"[[max_total_threads_per_threadgroup({})]] [[kernel]] {return_ty}"#,
            abi.cube_dim.num_elems(),
        )
    } else {
        return_ty
    };

    let entry_block = op.get_entry_block(ctx);

    let block = entry_block.deref(ctx);
    let params = block.arguments().enumerate();
    let params = params.map(|(i, arg)| gen_param(ctx, op, i, arg)).join(", ");

    let body = block_to_cpp(ctx, entry_block);

    format!("{attributes} {func_name}({params}) {{\n{body}\n}}\n")
});

fn gen_param(ctx: &Context, func: &FuncOp, i: usize, arg: Value) -> String {
    let mut segments = vec![];
    segments.push(arg.get_type(ctx).to_cpp(ctx));
    segments.push("const".into());
    segments.push(arg.name(ctx).to_string());
    if let Some(binding) = func.get_arg_attr::<BufferBindingAttr>(ctx, i, &ATTR_BUFFER_BINDING) {
        segments.push(format!("[[buffer({})]]", binding.buffer_pos));
    }
    if let Some(builtin) = func.get_arg_attr::<BuiltInAttr>(ctx, i, &ATTR_BUILTIN_ATTRIBUTE) {
        segments.push(format!("[[{}]]", builtin));
    }
    segments.join(" ")
}

// Metal does support dynamically sized shared memory, but it can't be used from WGPU. Metal allows
// allocating the full size statically, unlike CUDA, so it should be fine.
metal_op!(AllocSharedOp, |op, ctx| {
    let name = op.get_result(ctx).name(ctx);
    let align = op.alignment(ctx).0;
    let size = op.size(ctx).0;
    format!("alignas({align}) threadgroup char {name}[{size}];\n")
});

metal_op!(LoadInfoOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    let out = op.get_result(ctx);
    let out_ty = out.get_type(ctx).to_cpp(ctx);
    format!("constant {out_ty}& {} = *{ptr};\n", out.name(ctx))
});
