//! Aliasing and access attributes on a GPU kernel's buffer parameters.

use crate::prelude::BufferIOAttr;

/// Marks every pointer parameter `noalias`, and the read-only buffers and the metadata
/// `readonly`, from the recorded access modes. Distinct bindings never overlap, which is
/// what lets the backend keep a value in a register across a store, and a read-only pointer is
/// what lets NVPTX load through the non-coherent cache and AMDGPU prove a uniform load is not
/// clobbered and make it a scalar load.
///
/// # Safety
/// `func` must be a live function in `ctx` whose parameters are the buffers in binding order
/// followed by `metadata_params` metadata parameters.
pub(crate) unsafe fn annotate_buffer_params(
    ctx: llvm_sys::prelude::LLVMContextRef,
    func: llvm_sys::prelude::LLVMValueRef,
    io: &[BufferIOAttr],
    metadata_params: u32,
) {
    use llvm_sys::LLVMTypeKind;
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMCountParams, LLVMCreateEnumAttribute,
        LLVMGetEnumAttributeKindForName, LLVMGetParam, LLVMGetTypeKind, LLVMTypeOf,
    };

    unsafe {
        let enum_attr = |index: u32, name: &str| {
            let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
            if kind == 0 {
                return;
            }
            let attribute = LLVMCreateEnumAttribute(ctx, kind, 0);
            LLVMAddAttributeAtIndex(func, index, attribute);
        };

        // Atomic loads must retain coherent memory access.
        let may_say_readonly = !reads_atomically(func);

        let params = LLVMCountParams(func);
        let first_metadata = params.saturating_sub(metadata_params);
        for param in 0..params {
            if LLVMGetTypeKind(LLVMTypeOf(LLVMGetParam(func, param)))
                != LLVMTypeKind::LLVMPointerTypeKind
            {
                continue;
            }

            let index = param + 1;
            enum_attr(index, "noalias");

            let read_only = param >= first_metadata
                || io
                    .get(param as usize)
                    .is_some_and(|attr| *attr == BufferIOAttr::ReadOnly);
            if read_only && may_say_readonly {
                enum_attr(index, "readonly");
            }
        }
    }
}

/// # Safety
/// `func` must be a live LLVM function.
unsafe fn reads_atomically(func: llvm_sys::prelude::LLVMValueRef) -> bool {
    use llvm_sys::LLVMAtomicOrdering;
    use llvm_sys::core::{
        LLVMGetFirstBasicBlock, LLVMGetFirstInstruction, LLVMGetNextBasicBlock,
        LLVMGetNextInstruction, LLVMGetOrdering, LLVMIsALoadInst,
    };

    unsafe {
        let mut block = LLVMGetFirstBasicBlock(func);
        while !block.is_null() {
            let mut inst = LLVMGetFirstInstruction(block);
            while !inst.is_null() {
                if !LLVMIsALoadInst(inst).is_null()
                    && LLVMGetOrdering(inst) != LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic
                {
                    return true;
                }
                inst = LLVMGetNextInstruction(inst);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        false
    }
}
