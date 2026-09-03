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

use crate::nvptx::builtins::InsertNvptxBuiltinsPass;
use crate::shared::lowering::TargetLowering;
use crate::shared::metadata::{EntryArgLayout, rebuild_func_type};
use crate::shared::shared_memory::SharedDeclarations;

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

        // The arguments already carry the generic pointer type the rest of the pipeline works
        // with, and that is the type the entry keeps, so nothing is retyped here. The function
        // type is still rebuilt: the shared half of the entry ABI appended the `%info`
        // pointer, and the signature has to grow with it.
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
