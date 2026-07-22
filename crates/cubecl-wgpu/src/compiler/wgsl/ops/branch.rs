use cubecl_ir::dialect::branch::*;
use itertools::Itertools;

use crate::compiler::wgsl::{block_to_wgsl, to_wgsl::wgsl_op, value::WgslValue};

wgsl_op!(YieldOp, |_, _| String::new());

wgsl_op!(UnreachableOp, |_, _| "return;\n".into());
wgsl_op!(ReturnOp, |op, ctx| {
    match op.value(ctx) {
        Some(value) => format!("return {};\n", value.name(ctx)),
        None => "return;\n".into(),
    }
});

wgsl_op!(IfOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let then = block_to_wgsl(ctx, op.then_block(ctx));
    let else_ = block_to_wgsl(ctx, op.else_block(ctx));
    match else_.is_empty() {
        true => format!("if {cond} {{\n{then}\n}}\n"),
        false => format!("if {cond} {{\n{then}\n}} else {{\n{else_}\n}}\n"),
    }
});

wgsl_op!(SwitchOp, |op, ctx| {
    let val = op.value(ctx).name(ctx);
    let default = block_to_wgsl(ctx, op.default_block(ctx));
    let mut cases = op.cases(ctx).into_iter().map(|(val, block)| {
        let case = val.value().to_i128();
        format!("case {case}: {{\n{}\n}}", block_to_wgsl(ctx, block))
    });
    format!(
        "
switch({val}) {{
    {}
    default: {{
        {default}
    }}
}}\n",
        cases.join("\n")
    )
});

wgsl_op!(RangeLoopOp, |op, ctx| {
    let i = op.iter_var(ctx).name(ctx);
    let start = op.start(ctx).name(ctx);
    let end = op.end(ctx).name(ctx);
    let step = op.step(ctx).name(ctx);
    let body = block_to_wgsl(ctx, op.loop_body(ctx));
    format!("for(*{i} = {start}; *{i} < {end}; *{i} += {step}) {{\n{body}\n}}\n")
});

wgsl_op!(WhileOp, |op, ctx| {
    let cond = op.cond_ptr(ctx).name(ctx);
    let body = block_to_wgsl(ctx, op.loop_body(ctx));
    format!("while(*{cond}) {{\n{body}\n}}\n")
});
