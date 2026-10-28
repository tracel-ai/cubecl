//! `tile.drained_into`: an accumulator written out to a destination tile.

use cubecl::ir::{OpInserter, Scope};
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, dialect::branch::YieldOp,
    interfaces::LowerInterface, prelude::*,
};
use pliron::{
    basic_block::BasicBlock, derive::op_interface_impl, irbuild::inserter::OpInsertionPoint,
    linked_list::ContainsLinkedList,
};

/// An accumulator written out to a destination tile: its sums, scaled and cast, stored where the
/// destination says.
///
/// Its region holds the drain as tracing wrote it, and yields nothing: what it does is the writes.
/// Its lowering inlines the region; a backend that drains a tile another way lowers it its own way.
#[pliron_op(name = "tile.drained_into", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct DrainOp;

impl DrainOp {
    /// The drain `body` traces, inside a `tile.drained_into` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
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

        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        block.insert_at_front(op.deref(ctx).get_region(0), ctx);
        scope.register(&DrainOp { op });
    }
}

/// The code the region holds, in the op's place: what tracing wrote before the op existed.
#[op_interface_impl]
impl LowerInterface for DrainOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let op = self.get_operation();
        super::assert_nothing_escapes(ctx, op, "tile.drained_into");
        let block = self.get_body(ctx, 0);
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        let (_yield, body) = ops.split_last().expect("a drain's region ends in a yield");
        for &inner in body {
            rewriter.move_operation(ctx, inner, OpInsertionPoint::BeforeOperation(op));
        }
        rewriter.erase_operation(ctx, op);
        Ok(())
    }
}
