// C bindings for linking required bitcode definitions.

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

char *owned(const std::string &message) {
  char *copy = static_cast<char *>(std::malloc(message.size() + 1));
  if (copy != nullptr) {
    std::memcpy(copy, message.c_str(), message.size() + 1);
  }
  return copy;
}

} // namespace

/// Returns null on success or an owned error message. Free it with
/// `cubecl_free_message`.
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

  if (llvm::Linker::linkModules(module, std::move(*parsed),
                                llvm::Linker::Flags::LinkOnlyNeeded)) {
    return owned("linking device bitcode failed; see stderr above");
  }
  return nullptr;
}

extern "C" void cubecl_free_message(char *message) { std::free(message); }
