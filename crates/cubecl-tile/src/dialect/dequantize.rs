//! `tile.dequantize`: a tile's numbers written into another with its scales applied.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

/// A tile's numbers written into another with every scale it carries applied and its table looked
/// up: decoded from memory, or drained from an accumulator.
///
/// Its region holds the decode as tracing wrote it, and yields nothing: what it does is the writes
/// into the destination. Its lowering inlines the region; a backend that decodes tiles another way
/// lowers it its own way.
#[pliron_op(name = "tile.dequantize", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct DequantizeOp;

impl DequantizeOp {
    /// The decode `body` traces, inside a `tile.dequantize` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&DequantizeOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for DequantizeOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.dequantize");
        Ok(())
    }
}
