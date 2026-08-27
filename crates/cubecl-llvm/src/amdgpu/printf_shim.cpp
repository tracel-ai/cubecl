// C ABI over LLVM's AMDGPU printf emitter.
//
// A GPU has no `printf` to call, so the call has to become a conversation with
// the host. LLVM knows both halves of that conversation, which is
// `__ockl_printf_begin`, a string, some arguments widened to 64 bits, and a
// flag on the last one and clang reaches it through `emitAMDGPUPrintfCall`. It
// has no C entry point, so it gets one here rather than have this crate keep
// its own copy of an ABI it does not own.

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

  // `printf` answers in `i32` and the sequence in `i64`, so the result is
  // narrowed back before it replaces the call. Every caller so far drops it,
  // but a truncated count is still the honest answer rather than a type the
  // uses cannot take.
  if (!call->use_empty()) {
    llvm::Value *narrowed = builder.CreateTrunc(result, call->getType());
    call->replaceAllUsesWith(narrowed);
  }
  call->eraseFromParent();
}
