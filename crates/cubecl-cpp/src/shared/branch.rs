use cubecl_core::ir::{
    dialect::{branch::*, general::SelectOp},
    prelude::*,
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList};

use crate::shared::{CppValue, OpExtCPP, shared_op, shared_op_with_out, unroll::unrolling};

pub fn block_to_cpp(ctx: &Context, block: Ptr<BasicBlock>) -> String {
    let mut out = String::new();
    let ops = block.deref(ctx).iter(ctx);
    for op in ops {
        out.push_str(&op.to_cpp(ctx).unwrap());
    }
    out
}

shared_op!(ExecuteRegionOp, |op, ctx| {
    format!("{{{}}}", block_to_cpp(ctx, op.block(ctx)))
});

shared_op!(IfOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let else_block = op.else_block(ctx);
    let mut out = format!("if({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.then_block(ctx)));
    if else_block.deref(ctx).iter(ctx).count() > 1 {
        out.push_str("}\n else {\n");
        out.push_str(&block_to_cpp(ctx, else_block));
    }
    out.push_str("}\n");
    out
});

shared_op!(SwitchOp, |op, ctx| {
    let value = op.value(ctx).name(ctx);
    let mut out = format!("switch({value}) {{\n");
    for (value, block) in op.cases(ctx) {
        let block = block_to_cpp(ctx, block);
        let case = format!("case {}: {{ {block} break; }}\n", value.as_i128());
        out.push_str(&case);
    }
    let block = block_to_cpp(ctx, op.default_block(ctx));
    out.push_str(&format!("default: {{ {block} break; }}\n"));
    out.push_str("}\n");
    out
});

// Only relevant for IR structure
shared_op!(YieldOp, |_, _| String::new());

shared_op!(ReturnOp, |op, ctx| {
    if let Some(value) = op.value(ctx) {
        format!("return {};", value.name(ctx))
    } else {
        "return;".into()
    }
});

shared_op!(BreakOp, |_, _| "break;".into());
shared_op!(UnreachableOp, |_, _| "__builtin_unreachable();".into());

shared_op!(RangeLoopOp, |op, ctx| {
    let i = op.iter_var(ctx).name(ctx);
    let start = op.start(ctx).name(ctx);
    let end = op.end(ctx).name(ctx);
    let step = op.step(ctx).name(ctx);
    let mut out = format!("for(*{i} = {start}; *{i} < {end}; *{i} += {step}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op!(WhileOp, |op, ctx| {
    let cond = op.cond_ptr(ctx).name(ctx);
    let mut out = format!("while(*{cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op!(LoopOp, |op, ctx| {
    let mut out = "while(true) {\n".to_string();
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op_with_out!(SelectOp, |op, ctx| {
    let cond = op.condition(ctx).name(ctx);
    let then = op.true_value(ctx).name(ctx);
    let or_else = op.false_value(ctx).name(ctx);
    format!("{} ? {} : {}", cond, then, or_else)
});
unrolling!(SelectOp);
