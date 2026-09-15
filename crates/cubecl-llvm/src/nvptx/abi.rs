//! PTX kernel arguments.

use cubecl_core::ir::prelude::*;
use pliron::builtin::ops::FuncOp;

use cubecl_opt::passes::alloc_shared_memory::AllocateSharedMemoryBlockPass;
use pliron::pass::{OpPass, Passes};

use pliron_llvm::types::PointerType as LlvmPointerType;

use crate::nvptx::builtins::InsertNvptxBuiltinsPass;
use crate::shared::lowering::TargetLowering;
use crate::shared::metadata::{CtxGridConstants, EntryArgLayout, rebuild_func_type};
use crate::shared::shared_memory::SharedDeclarations;

const GLOBAL_ADDRESS_SPACE: u32 = 1;

#[derive(Debug, Default)]
pub struct PtxKernelParams;

impl EntryArgLayout for PtxKernelParams {
    fn present_args(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        buffers: &[(usize, usize, Value)],
        shared: SharedDeclarations,
    ) {
        debug_assert!(
            shared.is_empty(),
            "shared memory should have been lowered to the shared block by \
             AllocateSharedMemoryBlockPass"
        );

        // Kernel arguments must follow buffer binding order.
        debug_assert!(
            buffers
                .iter()
                .enumerate()
                .all(|(n, (_, buffer_pos, _))| n == *buffer_pos),
            "buffer arguments are not in binding order: {:?}",
            buffers.iter().map(|(i, p, _)| (*i, *p)).collect::<Vec<_>>()
        );

        let global_ptr = LlvmPointerType::get(ctx, GLOBAL_ADDRESS_SPACE).into();
        let entry = func.get_entry_block(ctx);
        for (arg_idx, _, _) in buffers {
            let arg = entry.deref(ctx).get_argument(*arg_idx);
            arg.set_type(ctx, global_ptr);
        }
        // By-value metadata stays in the generic address space.
        let last = entry.deref(ctx).get_num_arguments() - 1;
        if !ctx.grid_constants() {
            let info_arg = entry.deref(ctx).get_argument(last);
            info_arg.set_type(ctx, global_ptr);
        } else if last > buffers.len() {
            let dyn_meta_arg = entry.deref(ctx).get_argument(last - 1);
            dyn_meta_arg.set_type(ctx, global_ptr);
        }

        rebuild_func_type(ctx, func);
    }
}

pub struct NvptxLowering {
    /// Device warp width.
    pub plane_dim: u32,
}

impl TargetLowering for NvptxLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(AllocateSharedMemoryBlockPass);
    }

    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(InsertNvptxBuiltinsPass {
            plane_dim: self.plane_dim,
        });
    }

    fn arg_layout(&self) -> Box<dyn EntryArgLayout> {
        Box::new(PtxKernelParams)
    }
}
