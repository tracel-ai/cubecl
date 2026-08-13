use cubecl_core::ir::{
    dialect::{branch::*, general::SelectOp},
    prelude::*,
};
use pliron::{basic_block::BasicBlock, linked_list::ContainsLinkedList};

use crate::{
    error::EmissionErrors,
    shared::{
        CppValue, OpExtCPP, scoped_block, shared_op, shared_op_with_out, ty::TypeExtCPP,
        unroll::unrolling,
    },
};

pub fn block_to_cpp(ctx: &Context, block: Ptr<BasicBlock>) -> String {
    let mut out = String::new();
    let ops = block.deref(ctx).iter(ctx);
    for op in ops {
        // Emission runs under `Display`, which cannot fail, so an op with no lowering is
        // recorded and skipped rather than panicked on. `compile_ir` reads the record back and
        // fails the compilation; see [`EmissionErrors`] for why a panic here goes unnoticed.
        match op.to_cpp(ctx) {
            Ok(cpp) => out.push_str(&cpp),
            Err(err) => ctx.aux_ty::<EmissionErrors>().record(err),
        }
    }
    out
}

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
        let case = format!("case {}: {{ {block} break; }}\n", value.value().to_i128());
        out.push_str(&case);
    }
    let block = block_to_cpp(ctx, op.default_block(ctx));
    out.push_str(&format!("default: {{ {block} break; }}\n"));
    out.push_str("}\n");
    out
});

// Only relevant for IR structure
shared_op!(YieldOp, |_, _| String::new());
shared_op!(ConditionOp, |op, ctx| {
    format!("return {};", op.condition(ctx).name(ctx))
});

shared_op!(ReturnOp, |op, ctx| {
    if let Some(value) = op.value(ctx) {
        format!("return {};", value.name(ctx))
    } else {
        "return;".into()
    }
});

shared_op!(UnreachableOp, |_, _| "__builtin_unreachable();".into());

shared_op!(RangeLoopOp, |op, ctx| {
    let i = op.iter_var(ctx).name(ctx);
    let i_ty = op.iter_var(ctx).get_type(ctx).to_cpp(ctx);
    let start = op.start(ctx).name(ctx);
    let end = op.end(ctx).name(ctx);
    let step = op.step(ctx).name(ctx);
    let mut out = format!("for({i_ty} {i} = {start}; {i} < {end}; {i} += {step}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.loop_body(ctx)));
    out.push_str("}\n");
    out
});

shared_op!(WhileOp, |op, ctx| {
    let cond = scoped_block! {
        block_to_cpp(ctx, op.before_block(ctx))
    };
    let mut out = format!("while({cond}) {{\n");
    out.push_str(&block_to_cpp(ctx, op.after_block(ctx)));
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
