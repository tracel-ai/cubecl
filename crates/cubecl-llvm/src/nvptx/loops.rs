//! NVPTX loop hints.

use crate::{prelude::*, shared::branch::LoopShape};
use pliron_llvm::metadata::{
    MdNodeAttr, MdOperandAttr, attach_metadata, find_enclosing_module, get_metadata_table,
    set_metadata_table,
};

/// Asks LLVM to unroll a loop completely, within its pragma threshold.
const UNROLL_FULL: &str = "llvm.loop.unroll.full";

/// Marks the loop `latch` closes for the unrolling its `shape` calls for.
///
/// A loop of a constant trip count over a local array is unrolled completely, as NVVM does: only
/// then is every index a constant, and a local array indexed by constants alone becomes
/// registers instead of local memory. LLVM's cost model stops short of that for a loop of
/// more than a handful of steps (a top-k accumulator's 64, say), and the array it leaves in
/// local memory costs several times the loop. AMDGPU needs no hint: its cost model already
/// raises the unroll threshold for a loop that touches a private array.
pub fn mark(ctx: &Context, latch: Ptr<Operation>, shape: LoopShape) {
    match shape {
        LoopShape::ConstantOverLocalArray => add_loop_property(ctx, latch, UNROLL_FULL),
        LoopShape::Other => {}
    }
}

/// Attaches a loop id carrying `property` to `latch`, a loop's back edge.
fn add_loop_property(ctx: &Context, latch: Ptr<Operation>, property: &str) {
    let module = find_enclosing_module(ctx, latch).expect("a loop is inside a module");
    let mut table = get_metadata_table(ctx, module).unwrap_or_default();
    let property = table.push_uniqued(MdNodeAttr::new_tuple(vec![MdOperandAttr::String(
        property.to_string(),
    )]));
    // A loop id is a distinct node whose first operand is itself.
    let loop_id = table.reserve();
    table.set(
        loop_id,
        MdNodeAttr::new_distinct_tuple(vec![
            MdOperandAttr::Node(loop_id),
            MdOperandAttr::Node(property),
        ]),
    );
    set_metadata_table(ctx, module, table);
    attach_metadata(ctx, latch, "llvm.loop", loop_id);
}
