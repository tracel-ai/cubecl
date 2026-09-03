//! The PTX kernel parameter layout.
//!
//! Buffers stay as individual pointer arguments in binding order, with the metadata pointer
//! last. `CudaServer::execute` pushes resources in exactly that order and `CudaContext::
//! execute_task` hands them to `cuLaunchKernel` as `kernelParams`.
//!
//! Unlike the AMDGPU layout, the pointers are left in the generic address space rather than
//! retyped into the global one. NVPTX has a pass of its own for this — `NVPTXLowerArgs` infers
//! which kernel parameters point into global memory and rewrites the accesses — and it sees
//! more than a retype here could: a pointer that flows into a `getelementptr` chain before it
//! is dereferenced still gets inferred, where a blanket retype would only move the
//! `addrspacecast` a few instructions earlier. What the two share is the ordering contract,
//! which is the part the host depends on.

use cubecl_core::ir::prelude::*;
use pliron::builtin::ops::FuncOp;

use cubecl_opt::passes::alloc_shared_memory::AllocateSharedMemoryBlockPass;
use pliron::pass::{OpPass, Passes};

use pliron_llvm::types::PointerType as LlvmPointerType;

use crate::nvptx::builtins::InsertNvptxBuiltinsPass;
use crate::shared::lowering::TargetLowering;
use crate::shared::metadata::{EntryArgLayout, rebuild_func_type};
use crate::shared::shared_memory::SharedDeclarations;

/// Address space 1 is NVPTX's global address space, where a kernel's buffers live.
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

        // Parameter slot N must be buffer N. Holds because `KernelBuilder` assigns
        // `buffer_pos` from a monotonic counter and pushes the argument in the same call. If
        // that stops being true, fail here rather than passing buffers to the wrong slots.
        debug_assert!(
            buffers
                .iter()
                .enumerate()
                .all(|(n, (_, buffer_pos, _))| n == *buffer_pos),
            "buffer arguments are not in binding order: {:?}",
            buffers.iter().map(|(i, p, _)| (*i, *p)).collect::<Vec<_>>()
        );

        // Retype the buffers, and the `%info` pointer the shared half appended, into the
        // global address space.
        //
        // Not cosmetic: `NVPTXTagInvariantLoads` only marks a load invariant -- which is the
        // whole of how a load reaches the read-only cache as `ld.global.nc` -- when its
        // pointer is already in the global space *in the IR*, and inference runs too late to
        // give it that. Leaving them generic also leaves every access to be proven global
        // again downstream. `InferAddressSpaces` folds away the casts this leaves behind.
        //
        // Bind each argument before calling `set_type`: `get_argument` holds a `Ref` on the
        // entry block and `set_type` re-borrows it mutably, so chaining the two keeps the
        // guard alive across the statement and panics with "RefCell already borrowed".
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

/// The NVPTX target's contribution to the pipeline.
///
/// The hardware *is* the launch grid, so nothing is emulated: the shared memories are packed
/// into the one block a launch reserves, and the builtins become special register reads once
/// the polyfills that read them have been expanded.
pub struct NvptxLowering {
    /// Warp width of the device, which `PlaneDim` resolves to.
    pub plane_dim: u32,
}

impl TargetLowering for NvptxLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        // Packs every shared memory into one block of offsets, which the NVPTX lowering then
        // gives an address in `.shared`. Same pass the C++ backends and AMDGPU run.
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
