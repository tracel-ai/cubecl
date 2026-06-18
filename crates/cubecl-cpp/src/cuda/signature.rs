use cubecl_core::ir::{
    attributes::{EntrypointInterface, FuncInterface},
    settings::Dim3,
};
use itertools::Itertools;
use pliron::{
    builtin::{
        op_interfaces::SymbolOpInterface, ops::FuncOp, type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    context::Context,
    dict_key,
    r#type::Typed,
    value::Value,
};

use crate::{
    cuda::cuda_op,
    shared::{ATTR_CONST, CppValue, branch::block_to_cpp, kernel::ATTR_RESTRICT, ty::TypeExtCPP},
};

dict_key!(ATTR_GRID_CONST, "grid_const");

cuda_op!(FuncOp, |op, ctx| {
    let func_name = op.get_symbol_name(ctx);
    let ty = op.get_type(ctx).deref(ctx);
    let func_ty = ty.downcast_ref::<FunctionType>().unwrap();
    let return_ty = func_ty.res_types()[0].to_cpp(ctx);
    let attributes = if let Some(abi) = op.get_entrypoint_abi(ctx) {
        let cluster_dim = match abi.cluster_dim {
            Some(Dim3 { x, y, z }) => format!("__cluster_dims__(({x}, {y}, {z}))"),
            None => "".into(),
        };
        format!(
            r#"extern "C" __global__ {return_ty} __launch_bounds__({}) {cluster_dim}"#,
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
    if func.get_arg_attr(ctx, i, &ATTR_GRID_CONST).is_some() {
        segments.push("__grid_constant__".into());
    }
    segments.push(arg.get_type(ctx).to_cpp(ctx));
    if func.get_arg_attr(ctx, i, &ATTR_RESTRICT).is_some() {
        segments.push("__restrict__".into());
    }
    segments.push(arg.name(ctx).to_string());
    segments.join(" ")
}
