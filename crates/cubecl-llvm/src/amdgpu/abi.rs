//! The AMDGPU kernarg layout.
//!
//! Buffers stay as individual `ptr addrspace(1)` arguments in binding order, with
//! the metadata pointer last. `HipServer::execute` pushes resources
//! in exactly that order and `HipContext::execute_task` hands them to
//! `hipModuleLaunchKernel` as `kernelParams`

use cubecl_core::ir::prelude::*;
use pliron::builtin::ops::FuncOp;
use pliron_llvm::types::PointerType as LlvmPointerType;

use cubecl_opt::passes::alloc_shared_memory::AllocateSharedMemoryBlockPass;
use pliron::pass::{OpPass, Passes};

use crate::amdgpu::builtins::InsertAmdgpuBuiltinsPass;
use crate::shared::lowering::TargetLowering;
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
        debug_assert!(
            shared.is_empty(),
            "shared memory should have been lowered to LDS by AllocateSharedMemoryBlockPass"
        );

        // Kernarg slot N must be buffer N. Holds because `KernelBuilder` assigns
        // `buffer_pos` from a monotonic counter and pushes the argument in the same
        // call. If that stops being true, fail here rather than passing buffers to
        // the wrong kernargs.
        debug_assert!(
            buffers
                .iter()
                .enumerate()
                .all(|(n, (_, buffer_pos, _))| n == *buffer_pos),
            "buffer arguments are not in binding order: {:?}",
            buffers.iter().map(|(i, p, _)| (*i, *p)).collect::<Vec<_>>()
        );

        // Retype the buffers, and the `%info` pointer the shared half appended, into
        // the global address space.
        //
        // Bind each argument before calling `set_type`: `get_argument` holds a `Ref`
        // on the entry block and `set_type` re-borrows it mutably, so chaining the
        // two keeps the guard alive across the statement and panics with
        // "RefCell already borrowed".
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

/// The AMDGPU target's contribution to the pipeline.
///
/// The hardware *is* the launch grid, so nothing is emulated: the shared memories are packed
/// into the one LDS block a launch reserves, and the builtins become intrinsic calls once the
/// polyfills that read them have been expanded.
pub struct AmdGpuLowering {
    /// Wavefront width of the device, which `PlaneDim` resolves to.
    pub plane_dim: u32,
}

impl TargetLowering for AmdGpuLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        // Packs every shared memory into one block of offsets, which the AMDGPU lowering then
        // gives an address in LDS. Same pass the C++ backends run.
        passes.add_pass(AllocateSharedMemoryBlockPass);
    }

    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(InsertAmdgpuBuiltinsPass {
            plane_dim: self.plane_dim,
        });
    }

    fn arg_layout(&self) -> Box<dyn EntryArgLayout> {
        Box::new(KernargArgs)
    }
}
