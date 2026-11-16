//! `tile.reduce`: a tile reduced into another across the axes it does not span.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

/// A tile reduced into another across the axes the destination does not span, under a monoid, from
/// the monoid's identity or onto what the destination holds.
///
/// Its region holds the reduction as tracing wrote it, and yields nothing: what it does is the
/// writes into the destination. Its lowering inlines the region; a backend that reduces tiles
/// another way lowers it its own way.
#[pliron_op(name = "tile.reduce", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct ReduceOp;

impl ReduceOp {
    /// The reduction `body` traces, inside a `tile.reduce` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&ReduceOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for ReduceOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.reduce");
        Ok(())
    }
}
