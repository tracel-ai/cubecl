use cubecl_core::ir::{
    attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, EntrypointInterface, FuncInterface},
    prelude::*,
};
use cubecl_opt::passes::alloc_shared_memory::AllocSharedOp;
use derive_more::From;
use itertools::Itertools;
use pliron::{
    builtin::{ops::FuncOp, types::FunctionType},
    dict_key,
};

use crate::{
    metal::{BuiltInAttribute, metal_op},
    shared::{CppValue, branch::block_to_cpp, ty::TypeExtCPP},
};

#[pliron_attr(name = "msl.builtin", format, verifier = "succ")]
#[derive(new, From, Debug, PartialEq, Clone, Copy)]
pub struct BuiltinAttr(pub BuiltInAttribute);

dict_key!(ATTR_BUILTIN_ATTRIBUTE, "metal_builtin");

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
    if let Some(builtin) = func.get_arg_attr::<BuiltinAttr>(ctx, i, &ATTR_BUILTIN_ATTRIBUTE) {
        segments.push(format!("{}", builtin.0));
    }
    segments.join(" ")
}

// Metal does support dynamically sized shared memory, but it can't be used from WGPU. Metal allows
// allocating the full size statically, unlike CUDA, so it should be fine.
metal_op!(AllocSharedOp, |op, ctx| {
    let name = op.get_result(ctx).name(ctx);
    let align = op.alignment(ctx).0;
    let size = op.size(ctx).0;
    format!("threadgroup __align__({align}) char {name}[{size}];")
});
