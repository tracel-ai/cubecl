//! `tile.drained_into`: an accumulator written out to a destination tile.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

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
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&DrainOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for DrainOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.drained_into");
        Ok(())
    }
}
