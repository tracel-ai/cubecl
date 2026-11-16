//! `tile.mma`: two tiles contracted into an accumulator.

use cubecl::ir::Scope;
use cubecl_ir::{
    RecursiveMemoryEffects, RecursiveSideEffects, interfaces::LowerInterface, prelude::*,
};
use pliron::derive::op_interface_impl;

/// Two tiles contracted into an accumulator: `acc += lhs · rhs`, over the axes the accumulator
/// does not span.
///
/// Its region holds the contraction as tracing wrote it, and yields nothing: what it does is the
/// writes into the accumulator. Its lowering inlines the region; a backend that contracts tiles
/// another way lowers it its own way.
#[pliron_op(name = "tile.mma", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(RecursiveMemoryEffects, RecursiveSideEffects)]
pub struct MmaOp;

impl MmaOp {
    /// The contraction `body` traces, inside a `tile.mma` op.
    pub(crate) fn trace(scope: &Scope, body: impl FnOnce(&Scope)) {
        let op = super::trace_writes::<Self>(scope, body);
        scope.register(&MmaOp { op });
    }
}

#[op_interface_impl]
impl LowerInterface for MmaOp {
    fn lower(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        super::inline_writes(ctx, rewriter, self.get_operation(), "tile.mma");
        Ok(())
    }
}
