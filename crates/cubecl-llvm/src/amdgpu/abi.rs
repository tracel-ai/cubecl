//! The AMDGPU kernarg layout.
//!
//! Buffers stay as individual `ptr addrspace(1)` kernel arguments in binding
//! order, with the metadata pointer appended last. That is not a free choice:
//! `HipServer::execute` pushes resources in exactly that order
//! (`crates/cubecl-hip/src/compute/server.rs:801-816`) and
//! `HipContext::execute_task` hands them to `hipModuleLaunchKernel` as
//! `kernelParams`, so this layout is what makes the existing launch path work
//! untouched.

use cubecl_core::ir::prelude::*;
use pliron::builtin::ops::FuncOp;
use pliron_llvm::types::PointerType as LlvmPointerType;

use crate::shared::metadata::{EntryArgLayout, rebuild_func_type};
use crate::shared::shared_memory::SharedDeclarations;

/// Address space 1 is the AMDGPU global address space.
const GLOBAL_ADDRESS_SPACE: u32 = 1;

#[derive(Debug, Default)]
pub struct KernargArgs;

impl EntryArgLayout for KernargArgs {
    fn present_args(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        buffers: &[(usize, usize, Value)],
        shared: SharedDeclarations,
    ) {
        if !shared.is_empty() {
            unimplemented!("shared memory is not supported on the AMDGPU target yet");
        }

        // `HipServer` pushes resources in `buffer_pos` order, so kernarg slot N must be
        // buffer N. That holds because `KernelBuilder` assigns `buffer_pos` from a
        // monotonic counter and pushes the argument in the same call
        // (`cubecl-core/src/compute/builder.rs:54-70`). If that ever stops being true,
        // fail here rather than passing buffers to the wrong kernargs.
        debug_assert!(
            buffers
                .iter()
                .enumerate()
                .all(|(n, (_, buffer_pos, _))| n == *buffer_pos),
            "buffer arguments are not in binding order: {:?}",
            buffers.iter().map(|(i, p, _)| (*i, *p)).collect::<Vec<_>>()
        );

        // Buffers are already arguments in binding order, so all that is left
        // is to retype them into the global address space. The `%info` pointer
        // was appended by the shared half of the pass and gets the same
        // treatment.
        let global_ptr = LlvmPointerType::get(ctx, GLOBAL_ADDRESS_SPACE).into();
        let entry = func.get_entry_block(ctx);
        for (arg_idx, _, _) in buffers {
            entry
                .deref(ctx)
                .get_argument(*arg_idx)
                .set_type(ctx, global_ptr);
        }
        let info_idx = entry.deref(ctx).get_num_arguments() - 1;
        entry
            .deref(ctx)
            .get_argument(info_idx)
            .set_type(ctx, global_ptr);

        rebuild_func_type(ctx, func);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `KernargArgs` must be usable as a boxed layout, which is how
    /// `LowerEntryAbiPass` stores it. This fails to compile if the trait's
    /// object safety is broken by a later signature change.
    #[test]
    fn kernarg_args_is_a_boxed_layout() {
        let layout: Box<dyn EntryArgLayout> = Box::new(KernargArgs);
        let _ = layout;
    }
}
