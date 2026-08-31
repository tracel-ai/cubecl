//! The CPU entry layout: one pointer table for every resource the host owns.
//!
//! Buffers and shared memories collapse behind a single `%buffer_ptrs` indirection, so the JIT
//! host calls every kernel through the one `extern "C"` signature in
//! [`jit::engine`](super::jit::engine).

use core::cell::RefCell;
use std::rc::Rc;

use cubecl_core::ir::prelude::*;
use pliron::basic_block::BasicBlock;
use pliron::builtin::ops::FuncOp;

use pliron::pass::{OpPass, Passes};

use crate::cpu::entrypoint::InsertConstantEmulationPass;
use crate::cpu::shared_memory::SharedMemories;
use crate::shared::lowering::TargetLowering;
use crate::shared::metadata::{EntryArgLayout, load_table, rebuild_func_type, table_ty};
use crate::shared::shared_memory::SharedDeclarations;

pub struct TableArgs {
    /// Filled in with the shared memory the host must reserve, see [`SharedMemories`].
    shared_memories: Rc<RefCell<SharedMemories>>,
}

impl TableArgs {
    pub fn new(shared_memories: Rc<RefCell<SharedMemories>>) -> Self {
        Self { shared_memories }
    }
}

impl EntryArgLayout for TableArgs {
    fn present_args(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        buffers: &[(usize, usize, Value)],
        shared: SharedDeclarations,
    ) {
        let entry = func.get_entry_block(ctx);

        let shared_base = (buffers.iter())
            .map(|(_, buffer_pos, _)| buffer_pos + 1)
            .max()
            .unwrap_or(0);
        if !buffers.is_empty() || !shared.is_empty() {
            let table_ty = table_ty(ctx);
            BasicBlock::insert_argument(entry, ctx, 0, table_ty);
            let buffer_ptrs = entry.deref(ctx).get_argument(0);
            let terminator = entry
                .deref(ctx)
                .get_terminator(ctx)
                .expect("entry block must be terminated");

            for (_idx, buffer_pos, old_val) in buffers.iter() {
                let buffer_ty = old_val.get_type(ctx);
                let buffer = load_table(ctx, buffer_ptrs, *buffer_pos, buffer_ty, terminator);
                old_val.replace_all_uses_with(ctx, &buffer);
            }

            let blocks = shared.lower(ctx, buffer_ptrs, shared_base, terminator);
            if !blocks.is_empty() {
                *self.shared_memories.borrow_mut() = SharedMemories {
                    base: shared_base,
                    blocks,
                };
            }

            let mut removed: Vec<usize> = buffers.iter().map(|(i, _, _)| i + 1).collect();
            removed.sort_unstable();
            for idx in removed.into_iter().rev() {
                BasicBlock::remove_argument(entry, ctx, idx);
            }
        }

        rebuild_func_type(ctx, func);
    }
}

/// The CPU target's contribution to the pipeline.
///
/// A CPU has no launch grid, so the whole of it is emulated: the entry point becomes a loop
/// nest over the cube, and the shared memories become slots in the pointer table above.
pub struct CpuLowering {
    shared_memories: Rc<RefCell<SharedMemories>>,
}

impl CpuLowering {
    pub fn new(shared_memories: Rc<RefCell<SharedMemories>>) -> Self {
        Self { shared_memories }
    }
}

impl TargetLowering for CpuLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(InsertConstantEmulationPass);
    }

    fn arg_layout(&self) -> Box<dyn EntryArgLayout> {
        Box::new(TableArgs::new(self.shared_memories.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LowerEntryAbiPass` stores the layout boxed, so this fails to compile if a later signature
    /// change breaks the trait's object safety.
    #[test]
    fn table_args_is_a_boxed_layout() {
        let layout: Box<dyn EntryArgLayout> = Box::new(TableArgs::new(Rc::default()));
        let _ = layout;
    }
}
