//! Test only: each `tile.level` op lowered to a plain loop over the level's instances, run by the
//! first instance alone. The kernel must then give the numbers the default lowering gives, which
//! shows a backend can run a level some other way than by reading its position in the launch.
//!
//! Every instance runs the whole block the op sits in, at its own position, with its own copy of
//! everything the block declares. So the first instance runs that block once at every position,
//! and the others skip it. That holds for a level whose instances never meet: a plane level whose
//! block never syncs the cube, as in a plane routine, and a cube level whose block never syncs the
//! device. A level whose instances meet is refused.

use cubecl::ir::{Builtin, ExpandValue, Scope, dialect::general::ReadBuiltinOp};
use cubecl::prelude::*;
use cubecl_ir::{
    AddressSpace,
    dialect::{
        memory::{DeclareVariableOp, StoreOp},
        synchronization::SyncScope,
    },
    interfaces::Synchronizes,
    prelude::*,
};
use pliron::{
    attribute::boxed_attr_cast,
    basic_block::BasicBlock,
    builtin::ops::ConstantOp,
    graph::walkers::uninterruptible::immutable::walk_op,
    irbuild::{inserter::OpInsertionPoint, listener::DummyListener},
    linked_list::ContainsLinkedList,
    verify_err_noloc,
};

use super::level::LevelOp;

#[derive(thiserror::Error, Debug)]
#[error(
    "the sequential level lowering runs a level's instances along CUBE_POS_X, CUBE_POS_Y, \
     CUBE_POS_Z and UNIT_POS_Y only, and this level reads {0:?}"
)]
struct UnsupportedPosition(Builtin);

#[derive(thiserror::Error, Debug)]
#[error(
    "the sequential level lowering runs a level's instances one after the other, and this \
     level's instances meet at {0:?} scope, so the first would wait for the others"
)]
struct InstancesMeet(SyncScope);

/// The reads of the instance's position in the regions of the levels `block` holds, with the
/// position each reads.
fn position_reads(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Vec<(ReadBuiltinOp, Builtin)>> {
    let mut reads = Vec::new();
    let levels = block
        .deref(ctx)
        .iter(ctx)
        .filter(|op| op.dyn_op(ctx).downcast_ref::<LevelOp>().is_some());
    for level in levels {
        walk_op(
            ctx,
            &mut reads,
            &WALKCONFIG_PREORDER_FORWARD,
            level,
            |ctx, reads, node| {
                if let IRNode::Operation(op) = node
                    && let Some(read) = op.dyn_op(ctx).downcast_ref::<ReadBuiltinOp>()
                {
                    reads.push((*read, read.builtin(ctx).0));
                }
            },
        );
    }
    reads.retain(|(_, builtin)| !matches!(count_of(*builtin), Count::Not));
    match reads
        .iter()
        .find(|(_, builtin)| matches!(count_of(*builtin), Count::Unsupported))
    {
        Some((_, builtin)) => verify_err_noloc!(UnsupportedPosition(*builtin)),
        None => Ok(reads),
    }
}

/// What a builtin read inside a level counts among.
enum Count {
    /// A position the loop ranges over this count.
    Of(Builtin),
    /// A position this lowering has no loop for.
    Unsupported,
    /// Not a position.
    Not,
}

fn count_of(builtin: Builtin) -> Count {
    match builtin {
        Builtin::CubePosX => Count::Of(Builtin::CubeCountX),
        Builtin::CubePosY => Count::Of(Builtin::CubeCountY),
        Builtin::CubePosZ => Count::Of(Builtin::CubeCountZ),
        Builtin::UnitPosY => Count::Of(Builtin::CubeDimY),
        Builtin::UnitPos
        | Builtin::UnitPosX
        | Builtin::UnitPosZ
        | Builtin::UnitPosPlane
        | Builtin::PlanePos
        | Builtin::CubePos
        | Builtin::AbsolutePos
        | Builtin::AbsolutePosX
        | Builtin::AbsolutePosY
        | Builtin::AbsolutePosZ => Count::Unsupported,
        _ => Count::Not,
    }
}

/// Wraps the block holding `level` in `if position == 0 { for position in 0..count { .. } }`, one
/// per position the regions of the block's levels read, each read then standing for its loop's
/// index.
pub(super) fn run_on_first_instance(
    ctx: &mut Context,
    rewriter: &mut MatchRewriter,
    level: &LevelOp,
) -> Result<()> {
    let block = level
        .get_operation()
        .deref(ctx)
        .get_parent_block()
        .expect("a level sits in a block");
    // Every level in the block reads the same position, so they share one loop: the first of
    // them builds it, and the others find their reads gone.
    let reads = position_reads(ctx, block)?;
    if reads.is_empty() {
        return Ok(());
    }
    // A plane's instances meet at the cube, a cube's at the device.
    let meet = if reads
        .iter()
        .any(|(_, builtin)| *builtin == Builtin::UnitPosY)
    {
        SyncScope::Cube
    } else {
        SyncScope::Device
    };
    initialize_again(ctx, block);
    // Everything but the block's terminator, its last op.
    let mut ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
    ops.pop();
    for &inner in &ops {
        check_alone(ctx, inner, meet)?;
    }
    let mut positions = reads
        .iter()
        .filter_map(|(_, builtin)| match count_of(*builtin) {
            Count::Of(count) => Some((*builtin, count)),
            Count::Unsupported | Count::Not => None,
        })
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();

    rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(block));
    let scope = Scope::from_context_and_inserter(ctx, rewriter);
    let mut indices = Vec::new();
    nest(&scope, &positions, &mut indices, &mut |scope| {
        let mut at = scope.inserter().get_insertion_point();
        let mut mover = IRRewriter::<DummyListener>::default();
        for &inner in &ops {
            mover.move_operation(scope.ctx(), inner, at);
            at = OpInsertionPoint::AfterOperation(inner);
        }
        scope.inserter().set_insertion_point(at);
    });
    drop(scope);

    for (read, builtin) in reads {
        let at = positions.iter().position(|(p, _)| *p == builtin).unwrap();
        rewriter.replace_value_uses_with(ctx, read.get_result(ctx), indices[at]);
        rewriter.erase_operation(ctx, read.get_operation());
    }
    Ok(())
}

/// Stores each local variable `block` declares its initial value right after the declaration. A
/// declaration initializes once per kernel, and the block is about to run once per position.
fn initialize_again(ctx: &mut Context, block: Ptr<BasicBlock>) {
    let declarations = block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| op.dyn_op(ctx).downcast_ref::<DeclareVariableOp>().copied())
        .collect::<Vec<_>>();
    for declaration in declarations {
        if declaration.addr_space(ctx).0 != AddressSpace::Local {
            continue;
        }
        let Some(init) = declaration.initializer(ctx).map(|init| init.clone()) else {
            continue;
        };
        let constant = ConstantOp::new(ctx, boxed_attr_cast(init).unwrap());
        constant
            .get_operation()
            .insert_after(ctx, declaration.get_operation());
        let store = StoreOp::new(ctx, declaration.get_result(ctx), constant.get_result(ctx));
        store
            .get_operation()
            .insert_after(ctx, constant.get_operation());
    }
}

/// Fails where `root` syncs at `meet` scope or wider, or waits on a barrier, which the instances
/// running one after the other never reach together.
fn check_alone(ctx: &Context, root: Ptr<Operation>, meet: SyncScope) -> Result<()> {
    let mut widest = None;
    walk_op(
        ctx,
        &mut widest,
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, widest, node| {
            if let IRNode::Operation(op) = node {
                let scope = match op_cast::<dyn Synchronizes>(&*op.dyn_op(ctx)) {
                    Some(sync) => Some(sync.maximum_scope(ctx)),
                    None if Operation::get_opid(op, ctx).dialect.to_string() == "barrier" => {
                        Some(SyncScope::Cube)
                    }
                    None => None,
                };
                *widest = (*widest).max(scope);
            }
        },
    );
    match widest {
        Some(scope) if scope >= meet => verify_err_noloc!(InstancesMeet(scope)),
        _ => Ok(()),
    }
}

/// `if position == 0 { for index in 0..count { .. } }` for the first of `positions`, each with its
/// count, around the same for the others, around `body`.
fn nest(
    scope: &Scope,
    positions: &[(Builtin, Builtin)],
    indices: &mut Vec<Value>,
    body: &mut dyn FnMut(&Scope),
) {
    let Some((&(position, count), others)) = positions.split_first() else {
        body(scope);
        return;
    };
    let zero = NativeExpand::<u32>::from_lit(scope, 0u32);
    let first = read_builtin(scope, position).__expand_eq_method(scope, &zero);
    if_expand(scope, first, |scope| {
        let count = read_builtin(scope, count);
        RangeExpand::new(NativeExpand::from_lit(scope, 0u32), count).expand(
            scope,
            &mut |scope, index: NativeExpand<u32>| {
                indices.push(index.expand.value(scope));
                nest(scope, others, indices, body);
            },
        );
    });
}

fn read_builtin(scope: &Scope, builtin: Builtin) -> NativeExpand<u32> {
    let ty = u32::__expand_as_type(scope);
    let op = ReadBuiltinOp::new(scope.ctx_mut(), ty, builtin);
    scope.register(&op);
    ExpandValue::Value(op.get_result(scope.ctx())).into()
}
