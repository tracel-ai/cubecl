// C ABI over LLVM's bitcode linker.
//
// The C API links whole modules: `LLVMLinkModules2` pulls every definition of
// the source in and leaves the rest to be dead-stripped later. ROCm's device
// libraries are meant to be linked the way `llvm-link --only-needed` does,
// taking only the definitions the kernel actually calls

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include <llvm-c/Core.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

namespace {

/// A copy of `message` the Rust side can take and free with `free`.
char *owned(const std::string &message) {
  char *copy = static_cast<char *>(std::malloc(message.size() + 1));
  if (copy != nullptr) {
    std::memcpy(copy, message.c_str(), message.size() + 1);
  }
  return copy;
}

} // namespace

/// Links the definitions `dest` needs out of the bitcode in `[data, data +
/// len)`.
///
/// Returns null on success, else a `malloc`'d message the caller owns. The
/// bitcode is parsed into `dest`'s own context, as `llvm::Linker` requires.
extern "C" char *cubecl_link_device_bitcode(LLVMModuleRef dest,
                                            const char *data, size_t len) {
  llvm::Module &module = *llvm::unwrap(dest);

  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, len), "device-lib",
                                       /*RequiresNullTerminator=*/false);
  auto parsed =
      llvm::parseBitcodeFile(buffer->getMemBufferRef(), module.getContext());
  if (!parsed) {
    return owned("parsing device bitcode: " +
                 llvm::toString(parsed.takeError()));
  }

  // The source module's triple and layout are the device libraries' own, which
  // match the module `finalize_ir` stamped; a mismatch would only warn, so the
  // caller checks the arch instead.
  if (llvm::Linker::linkModules(module, std::move(*parsed),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    return owned("linking device bitcode failed; see stderr above");
  }
  return nullptr;
}

/// Frees what `cubecl_link_device_bitcode` returned, on the allocator that made
/// it.
extern "C" void cubecl_free_message(char *message) { std::free(message); }
