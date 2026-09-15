// C bindings for AMDGPU hostcall printing.

#include <cstddef>

#include <llvm-c/Core.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Transforms/Utils/AMDGPUEmitPrintf.h>

extern "C" void cubecl_emit_amdgpu_printf(LLVMValueRef call_ref) {
  auto *call = llvm::cast<llvm::CallInst>(llvm::unwrap(call_ref));

  llvm::SmallVector<llvm::Value *, 8> args(call->args());

  llvm::IRBuilder<> builder(call);
  llvm::Value *result =
      llvm::emitAMDGPUPrintfCall(builder, args, /*isBuffered=*/false);

  if (!call->use_empty()) {
    llvm::Value *narrowed = builder.CreateTrunc(result, call->getType());
    call->replaceAllUsesWith(narrowed);
  }
  call->eraseFromParent();
}
