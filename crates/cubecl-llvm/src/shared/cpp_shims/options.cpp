// C bindings for setting a registered LLVM command-line option.

#include <llvm/Support/CommandLine.h>

/// Sets the option `name` to `value` as if it had been passed on a command
/// line. Returns false when no such option is registered or it refuses the
/// value, where `LLVMParseCommandLineOptions` would exit the process instead.
extern "C" bool cubecl_set_llvm_option(const char *name, const char *value) {
  auto &options = llvm::cl::getRegisteredOptions();
  auto found = options.find(name);
  if (found == options.end()) {
    return false;
  }
  // `addOccurrence` returns true on error.
  return !found->second->addOccurrence(0, name, value);
}
