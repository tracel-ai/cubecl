// C ABI over LLVM's AMDGPU printf emitter, which has no C entry point of its
// own.
//
// `isBuffered` is false: the buffered form is the OpenCL one, writing a packed
// record for the runtime to match against an `amdhsa.printf` note. The hostcall
// form asks the host directly, and is what HIP emits.

#include <cstddef>

#include <llvm-c/Core.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Transforms/Utils/AMDGPUEmitPrintf.h>

/// Rewrites the `printf` call `call_ref` into the hostcall sequence, in place.
///
/// The first argument must be the format string, as it is for any `printf`. The
/// call's own result is replaced by what the sequence returns, and the call
/// erased.
extern "C" void cubecl_emit_amdgpu_printf(LLVMValueRef call_ref) {
  auto *call = llvm::cast<llvm::CallInst>(llvm::unwrap(call_ref));

  llvm::SmallVector<llvm::Value *, 8> args(call->args());

  llvm::IRBuilder<> builder(call);
  llvm::Value *result =
      llvm::emitAMDGPUPrintfCall(builder, args, /*isBuffered=*/false);

  // `printf` answers in `i32` and the sequence in `i64`.
  if (!call->use_empty()) {
    llvm::Value *narrowed = builder.CreateTrunc(result, call->getType());
    call->replaceAllUsesWith(narrowed);
  }
  call->eraseFromParent();
}
