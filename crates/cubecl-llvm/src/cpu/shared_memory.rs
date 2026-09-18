//! Host-managed shared memory for CPU kernels.

use crate::{
    prelude::*,
    shared::{metadata::load_table, shared_memory::SharedMemoryBlock},
};

/// Shared memory blocks and their pointer table offset.
#[derive(Clone, Debug, Default)]
pub struct SharedMemories {
    pub base: usize,
    pub blocks: Vec<SharedMemoryBlock>,
}

impl SharedDeclarations {
    /// `before` must dominate all uses of the shared memory pointers.
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
