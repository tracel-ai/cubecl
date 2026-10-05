//! AMDGPU kernel arguments.

use crate::{
    amdgpu::builtins::AmdGpuDispatch,
    prelude::*,
    shared::{builtins::InsertGpuBuiltinsPass, metadata::rebuild_func_type},
};
use cubecl_opt::passes::alloc_shared_memory::AllocateSharedMemoryBlockPass;

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
        debug_assert!(
            shared.is_empty(),
            "shared memory should have been lowered to LDS by AllocateSharedMemoryBlockPass"
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
        let info_idx = entry.deref(ctx).get_num_arguments() - 1;
        let info_arg = entry.deref(ctx).get_argument(info_idx);
        info_arg.set_type(ctx, global_ptr);

        rebuild_func_type(ctx, func);
    }
}

pub struct AmdGpuLowering {
    /// Device wavefront width.
    pub plane_dim: u32,
}

impl TargetLowering for AmdGpuLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(AllocateSharedMemoryBlockPass);
    }

    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(InsertGpuBuiltinsPass::new(
            Box::new(AmdGpuDispatch),
            self.plane_dim,
        ));
    }

    fn arg_layout(&self) -> Box<dyn EntryArgLayout> {
        Box::new(KernargArgs)
    }
}
