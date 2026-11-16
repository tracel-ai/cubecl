use cubecl_core::ir::{dialect::branch::*, prelude::*};

use crate::{
    opencl::opencl_op,
    shared::{CppValue, branch::block_to_cpp},
};

// C has no lambdas: the condition block runs at the top of an endless loop, and its condition
// leaves the loop.
opencl_op!(WhileOp, |op, ctx| {
    let mut out = "for (;;) {\n".to_string();
    out.push_str(&block_to_cpp(ctx, op.before_block(ctx)));
    out.push_str(&block_to_cpp(ctx, op.after_block(ctx)));
    out.push_str("}\n");
    out
});

opencl_op!(ConditionOp, |op, ctx| {
    format!("if (!{}) break;\n", op.condition(ctx).name(ctx))
});
