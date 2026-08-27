//! Lowering of the shared `memory.declare_variable` to a pointer the host provides.
//!
//! Every unit of a cube runs the kernel on a thread of its own, so shared memory cannot be a
//! stack allocation: all the units have to see the same bytes. The host reserves a block per
//! shared memory instead (out of the stream's shared memory pool) and passes it the same way as a
//! buffer, in the pointer table the entry ABI hands to the kernel. Declaring shared memory then
//! costs no more than reading a kernel argument.
//!
//! Like the units, the cubes of a launch share those blocks, which is what a cube barrier is for:
//! without one, a unit racing ahead to the next cube may overwrite what the units still finishing
//! the current one are reading.

use cubecl_core::ir::prelude::*;

use crate::shared::metadata::load_table;
use crate::shared::shared_memory::{SharedDeclarations, SharedMemoryBlock};

/// The shared memory of a kernel: the blocks to reserve, and the slot of the pointer table their
/// pointers go to. They sit right after the buffers, which own the slots before `base`.
#[derive(Clone, Debug, Default)]
pub struct SharedMemories {
    pub base: usize,
    pub blocks: Vec<SharedMemoryBlock>,
}

impl SharedDeclarations {
    /// Replaces every declaration by the pointer `table` holds for it, starting at slot `base`.
    /// The loads go before `before`, which must be in a block dominating every use.
    pub fn lower(
        self,
        ctx: &mut Context,
        table: Value,
        base: usize,
        before: Ptr<Operation>,
    ) -> Vec<SharedMemoryBlock> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(offset, (decl, result, block))| {
                let ptr_ty = result.get_type(ctx);
                let ptr = load_table(ctx, table, base + offset, ptr_ty, before);

                result.replace_all_uses_with(ctx, &ptr);
                Operation::erase(decl, ctx);
                block
            })
            .collect()
    }
}
