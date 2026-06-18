use cubecl_core::ir::{
    attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, EntrypointInterface, FuncInterface},
    prelude::*,
};
use derive_more::From;
use itertools::Itertools;
use pliron::{
    builtin::{ops::FuncOp, types::FunctionType},
    dict_key,
};

use crate::{
    metal::{BuiltInAttribute, metal_op},
    shared::{ATTR_CONST, CppValue, branch::block_to_cpp, ty::TypeExtCPP},
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
    if func.get_arg_attr(ctx, i, &ATTR_CONST).is_some() {
        segments.push("const".into());
    }
    segments.push(arg.get_type(ctx).to_cpp(ctx));
    segments.push(arg.name(ctx).to_string());
    if let Some(binding) = func.get_arg_attr(ctx, i, &ATTR_BUFFER_BINDING) {
        let binding = binding.downcast_ref::<BufferBindingAttr>().unwrap();
        segments.push(format!("[[buffer({})]]", binding.buffer_pos));
    }
    if let Some(builtin) = func.get_arg_attr(ctx, i, &ATTR_BUILTIN_ATTRIBUTE) {
        let builtin = builtin.downcast_ref::<BuiltinAttr>().unwrap().0;
        segments.push(format!("{builtin}"));
    }
    segments.join(" ")
}
