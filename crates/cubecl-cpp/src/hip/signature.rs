use cubecl_core::ir::{
    attributes::{EntrypointInterface, FuncInterface},
    prelude::*,
};
use itertools::Itertools;
use pliron::builtin::{ops::FuncOp, types::FunctionType};

use crate::{
    hip::hip_op,
    shared::{ATTR_CONST, ATTR_RESTRICT, CppValue, branch::block_to_cpp, ty::TypeExtCPP},
};

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
    if func.get_arg_attr(ctx, i, &ATTR_RESTRICT).is_some() {
        segments.push("__restrict__".into());
    }
    segments.push(arg.name(ctx).to_string());
    segments.join(" ")
}
