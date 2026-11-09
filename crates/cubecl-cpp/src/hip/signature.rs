use cubecl_core::ir::{
    attributes::{ATTR_PEER_BINDING, EntrypointInterface, FuncInterface},
    interfaces::TypedExt,
    prelude::*,
};
use itertools::Itertools;
use pliron::builtin::{
    ops::{FuncOp, ModuleOp},
    types::FunctionType,
};

use crate::{
    hip::{extension::compile_wmma_extensions, hip_op},
    shared::{
        CppValue,
        branch::block_to_cpp,
        define_array_polyfill,
        ty::{TypeExtCPP, TypedExtCPP},
        type_definitions,
    },
};

hip_op!(ModuleOp, |op, ctx| {
    let mut out = String::new();
    type_definitions(&mut out, "long long").unwrap();
    define_array_polyfill(&mut out).unwrap();
    out.push_str(&compile_wmma_extensions(ctx, op.get_operation()));
    out.push_str(&block_to_cpp(ctx, op.get_body(ctx, 0)));
    out
});

hip_op!(FuncOp, |op, ctx| {
    let func_name = op.get_symbol_name(ctx);
    let ty = op.get_type(ctx).deref(ctx);
    let func_ty = ty.downcast_ref::<FunctionType>().unwrap();
    let return_ty = func_ty.res_types()[0].to_cpp(ctx);
    let attributes = if let Some(abi) = op.get_entrypoint_abi(ctx) {
        format!(
            r#"extern "C" __global__ {return_ty} __launch_bounds__({})"#,
            abi.cube_dim.num_elems(),
        )
    } else {
        format!("__device__ {return_ty}")
    };

    let entry_block = op.get_entry_block(ctx);

    let block = entry_block.deref(ctx);
    let params = block.arguments().enumerate();
    // Another device may write a peer buffer while the kernel runs, so it is not promised
    // unaliased.
    let params = params
        .map(|(i, arg)| gen_param(ctx, arg, !op.has_arg_attr(ctx, i, &ATTR_PEER_BINDING)))
        .join(", ");

    let body = block_to_cpp(ctx, entry_block);

    format!("{attributes} {func_name}({params}) {{\n{body}\n}}\n")
});

fn gen_param(ctx: &Context, arg: Value, restrict: bool) -> String {
    let mut segments = vec![];
    segments.push(arg.get_type(ctx).to_cpp(ctx));
    segments.push("const".into());
    if restrict && (arg.is_ptr(ctx) || arg.is_uniform_ptr(ctx)) {
        segments.push("__restrict__".into());
    }
    segments.push(arg.name(ctx).to_string());
    segments.join(" ")
}
