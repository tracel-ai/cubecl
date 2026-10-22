//! `tile.level`: the tiles one instance takes at a level distributed over a scope.

use cubecl::ir::{ExpandValue, OpInserter, Scope};
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, RecursivelySpeculatable,
    dialect::branch::YieldOp, prelude::*,
};
use pliron::{
    basic_block::BasicBlock,
    graph::walkers::uninterruptible::immutable::walk_op,
    irbuild::{
        inserter::OpInsertionPoint,
        listener::DummyListener,
    },
    linked_list::ContainsLinkedList,
};

use crate::CoordsExpand;

/// The tiles one instance takes at a level distributed over a scope: which tile is whose.
///
/// Its region computes them from the instance's position in the launch, as tracing wrote it, and
/// yields the ones known only at runtime; its results stand for them. [`LowerLevelPass`] inlines
/// the region; a backend that runs the level's instances another way lowers it its own way.
#[pliron_op(
    name = "tile.level",
    format = "`: ` types(CharSpace(`,`)) ` ` region($0)",
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects, RecursivelySpeculatable)]
pub struct LevelOp;

impl LevelOp {
    /// The `(counts, positions, scales)` `body` traces, inside a `tile.level` op.
    pub(crate) fn trace(
        scope: &Scope,
        body: impl FnOnce(&Scope) -> TilesExpand,
    ) -> TilesExpand {
        let block = BasicBlock::new(scope.ctx_mut(), None, vec![]);
        let child = scope.branch_child(OpInserter::new_at_block_end(block));
        let (mut counts, mut positions, mut scales) = body(&child);

        let mut runtime = counts
            .runtime_mut()
            .chain(positions.runtime_mut())
            .chain(scales.runtime_mut())
            .collect::<Vec<_>>();
        let yielded = runtime
            .iter()
            .map(|v| v.expand.value(&child))
            .collect::<Vec<_>>();
        let ctx = scope.ctx_mut();
        let types = yielded.iter().map(|v| v.get_type(ctx)).collect();
        let yield_op = Operation::new(
            ctx,
            YieldOp::get_concrete_op_info(),
            vec![],
            yielded,
            vec![],
            0,
        );
        yield_op.insert_at_back(block, ctx);

        let op = Operation::new(ctx, Self::get_concrete_op_info(), types, vec![], vec![], 1);
        let region = op.deref(ctx).get_region(0);
        block.insert_at_front(region, ctx);
        let level = LevelOp { op };
        scope.register(&level);

        for (i, v) in runtime.iter_mut().enumerate() {
            **v = ExpandValue::Value(op.deref(scope.ctx()).get_result(i)).into();
        }
        drop(runtime);
        (counts, positions, scales)
    }
}

/// A walk's `(counts, positions, scales)`, as traced.
pub(crate) type TilesExpand = (
    CoordsExpand<usize>,
    CoordsExpand<usize>,
    CoordsExpand<usize>,
);

/// The `tile.level` ops under `root`, outer ones first.
pub(crate) fn levels(ctx: &Context, root: Ptr<Operation>) -> Vec<LevelOp> {
    let mut levels = Vec::new();
    walk_op(
        ctx,
        &mut levels,
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, levels, node| {
            if let IRNode::Operation(op) = node
                && let Some(level) = op.dyn_op(ctx).downcast_ref::<LevelOp>()
            {
                levels.push(*level);
            }
        },
    );
    levels
}

/// Lowers each `tile.level` op to the code its region holds, in its place: what tracing wrote
/// before the op existed.
#[derive(Default)]
pub struct LowerLevelPass;

#[pass_name]
impl Pass for LowerLevelPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let mut rewriter = IRRewriter::<DummyListener>::default();
        for level in levels(ctx, op) {
            res.ir_changed |= IRStatus::Changed;
            level.inline(ctx, &mut rewriter);
        }
        Ok(res)
    }
}

impl LevelOp {
    /// Moves the region's code before the op, and the values it yields in place of the results.
    pub(crate) fn inline(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) {
        let op = self.get_operation();
        let block = self.get_body(ctx, 0);
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        let (terminator, body) = ops.split_last().expect("a level's region ends in a yield");
        for &inner in body {
            rewriter.move_operation(ctx, inner, OpInsertionPoint::BeforeOperation(op));
        }
        let yielded = terminator.deref(ctx).operands().collect::<Vec<_>>();
        for (i, value) in yielded.into_iter().enumerate() {
            let result = op.deref(ctx).get_result(i);
            rewriter.replace_value_uses_with(ctx, result, value);
        }
        rewriter.erase_operation(ctx, op);
    }
}
