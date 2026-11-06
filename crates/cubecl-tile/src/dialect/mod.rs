//! The tile ops in the IR, each lowered by every backend's `LowerOpsPass`.

mod copy;
mod drain;
mod level;
mod logsumexp;
mod mma;
mod reduce;
#[cfg(feature = "sequential-levels")]
mod sequential;

pub use copy::CopyOp;
pub use drain::DrainOp;
pub use level::LevelOp;
pub use logsumexp::LogsumexpOp;
pub use mma::MmaOp;
pub use reduce::ReduceOp;

use cubecl::ir::{OpInserter, Scope};
use cubecl_ir::{dialect::branch::YieldOp, prelude::*};
use pliron::{
    basic_block::BasicBlock, irbuild::inserter::OpInsertionPoint, linked_list::ContainsLinkedList,
};

/// An op `T` with no operands and no results, whose one region holds what `body` traces:
/// the shape of a tile op that stands for writes alone. The caller registers it with the scope.
pub(crate) fn trace_writes<T: Op>(scope: &Scope, body: impl FnOnce(&Scope)) -> Ptr<Operation> {
    let block = BasicBlock::new(scope.ctx_mut(), None, vec![]);
    let child = scope.branch_child(OpInserter::new_at_block_end(block));
    body(&child);

    let ctx = scope.ctx_mut();
    let yield_op = Operation::new(
        ctx,
        YieldOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    yield_op.insert_at_back(block, ctx);

    let op = Operation::new(ctx, T::get_concrete_op_info(), vec![], vec![], vec![], 1);
    block.insert_at_front(op.deref(ctx).get_region(0), ctx);
    op
}

/// Lowers an op [`trace_writes`] built: the code its region holds, in its place, which is what
/// tracing wrote before the op existed.
pub(crate) fn inline_writes(
    ctx: &mut Context,
    rewriter: &mut MatchRewriter,
    op: Ptr<Operation>,
    name: &str,
) {
    assert_nothing_escapes(ctx, op, name);
    let block = op
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_head()
        .expect("a region with its block");
    let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
    let (_yield, body) = ops
        .split_last()
        .expect("a tile op's region ends in a yield");
    for &inner in body {
        rewriter.move_operation(ctx, inner, OpInsertionPoint::BeforeOperation(op));
    }
    rewriter.erase_operation(ctx, op);
}

/// Panics where a value `op`'s region defines is used outside it: an op that yields nothing
/// stands for writes alone, so what its region computes stays inside it, and a backend may lower
/// the op without keeping any of it.
pub(crate) fn assert_nothing_escapes(ctx: &Context, op: Ptr<Operation>, name: &str) {
    let inside = |user: Ptr<Operation>| {
        let mut at = Some(user);
        while let Some(current) = at {
            if current == op {
                return true;
            }
            at = current.deref(ctx).get_parent_op(ctx);
        }
        false
    };
    let mut pending = vec![op];
    while let Some(outer) = pending.pop() {
        let outer = outer.deref(ctx);
        for region in outer.regions() {
            for block in region.deref(ctx).iter(ctx) {
                for inner in block.deref(ctx).iter(ctx) {
                    for result in inner.deref(ctx).results() {
                        assert!(
                            result.uses(ctx).iter().all(|used| inside(used.user_op())),
                            "{name}: a value its region computes is used after it; the op yields \
                             nothing, so what it computes must stay inside it"
                        );
                    }
                    pending.push(inner);
                }
            }
        }
    }
}
