use alloc::{vec, vec::Vec};

use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::{DefiningEntity, Value},
};

use crate::{FuncOpExt, dialect::OperationPtrExt};

/// Lift a closure to a standalone function with the captures as extra args. Returns the value that
/// should be passed to these new args.
pub fn lift_closure(ctx: &Context, func: &FuncOp) -> Vec<Value> {
    let func_op = func.get_operation();
    let entry = func.get_entry_block(ctx);
    let mut captures = vec![];

    walk_op(
        ctx,
        &mut (func, &mut captures),
        &WALKCONFIG_PREORDER_FORWARD,
        func.get_operation(),
        |ctx, (func, captures), node| {
            if let IRNode::Operation(ptr) = node {
                for opd in ptr.operands(ctx) {
                    if !is_in_func(ctx, func, opd.defining_entity()) {
                        captures.push(opd);
                    }
                }
            }
        },
    );

    for capture in captures.iter() {
        let id = func.push_argument(ctx, capture.get_type(ctx));
        let arg_value = entry.deref(ctx).get_argument(id);
        capture.replace_some_uses_with(
            ctx,
            |ctx, r#use| op_is_in_func(ctx, func_op, r#use.user_op()),
            &arg_value,
        );
    }

    captures
}

fn op_is_in_func(ctx: &Context, func: Ptr<Operation>, mut op: Ptr<Operation>) -> bool {
    while let Some(parent) = op.deref(ctx).get_parent_op(ctx) {
        if parent == func {
            return true;
        }
        op = parent;
    }
    false
}

fn block_is_in_func(ctx: &Context, func: Ptr<BasicBlock>, mut block: Ptr<BasicBlock>) -> bool {
    if func == block {
        return true;
    }
    while let Some(parent) = block.deref(ctx).get_parent_block(ctx) {
        if parent == func {
            return true;
        }
        block = parent;
    }
    false
}

fn is_in_func(ctx: &Context, func: &FuncOp, entity: DefiningEntity) -> bool {
    match entity {
        DefiningEntity::Op(op) => op_is_in_func(ctx, func.get_operation(), op),
        DefiningEntity::Block(block) => block_is_in_func(ctx, func.get_entry_block(ctx), block),
    }
}
