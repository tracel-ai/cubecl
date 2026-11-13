//! `tile.logsumexp`: an online softmax's logsumexp written into a tile.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

/// An online softmax's logsumexp, `m + ln(l)` for each slice, written into a tile.
///
/// Its region holds the write as tracing wrote it, and yields nothing: what it does is the writes
/// into the tile. Its lowering inlines the region; a backend that writes it another way lowers it
/// its own way.
#[pliron_op(name = "tile.logsumexp", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct LogsumexpOp;

impl LogsumexpOp {
    /// The write `body` traces, inside a `tile.logsumexp` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&LogsumexpOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for LogsumexpOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.logsumexp");
        Ok(())
    }
}
