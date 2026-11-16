//! `tile.copy`: a tile's numbers moved into another as stored.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

/// A tile's numbers moved into another as they are stored, the pairing of the two kinds picking the
/// instruction: a load into a stage or into fragments, a store out of them.
///
/// Its region holds the copy as tracing wrote it, and yields nothing: what it does is the writes
/// into the destination. Its lowering inlines the region; a backend that moves tiles another way
/// lowers it its own way.
#[pliron_op(name = "tile.copy", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct CopyOp;

impl CopyOp {
    /// The copy `body` traces, inside a `tile.copy` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&CopyOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for CopyOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.copy");
        Ok(())
    }
}
