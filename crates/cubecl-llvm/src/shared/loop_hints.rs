//! Loop hints: what the lowering asks LLVM to do with a loop, as the metadata on its latch.

use crate::prelude::*;
use pliron_llvm::metadata::{
    MdNodeAttr, MdOperandAttr, attach_metadata, find_enclosing_module, get_metadata_table,
    set_metadata_table,
};

/// What a target asks LLVM for on a loop its cost model would decide differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopHint {
    /// Unroll the loop completely, within LLVM's pragma threshold.
    UnrollFull,
}

impl LoopHint {
    /// The `llvm.loop` property this hint is written as.
    fn property(self) -> &'static str {
        match self {
            LoopHint::UnrollFull => "llvm.loop.unroll.full",
        }
    }

    /// Attaches the hint to `latch`, the back edge of the loop it applies to.
    pub(crate) fn attach(self, ctx: &Context, latch: Ptr<Operation>) {
        let module = find_enclosing_module(ctx, latch).expect("a loop is inside a module");
        let mut table = get_metadata_table(ctx, module).unwrap_or_default();
        let property = table.push_uniqued(MdNodeAttr::new_tuple(vec![MdOperandAttr::String(
            self.property().to_string(),
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
}
